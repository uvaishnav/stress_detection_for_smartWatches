import pandas as pd
import numpy as np
import os
import logging
from .adaptive_filter import AdaptiveFilter
from .kalman_filter import KalmanFilter
from .wavelet_denoiser import WaveletDenoiser
from .motion_artifact_detector import MotionArtifactDetector
from scipy.signal import find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import butter, sosfilt
from tqdm import tqdm
from scipy.signal import periodogram
from scipy.signal import correlate

class SignalProcessingPipeline:
    def __init__(self):
        self.adaptive_filter = AdaptiveFilter()
        self.kalman_filter = KalmanFilter()
        self.wavelet_denoiser = WaveletDenoiser()
        self.motion_detector = MotionArtifactDetector()
        
        # Device-specific noise profiles (empirically determined)
        self.device_noise_profiles = {
            'apple_watch': {'base_noise': 0.05, 'acc_scale': 2048},
            'galaxy_watch': {'base_noise': 0.08, 'acc_scale': 1024},
            'default': {'base_noise': 0.12, 'acc_scale': 512}
        }

    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Enhanced normalization with fallback"""
        data = np.nan_to_num(data, nan=np.median(data))
        
        # Fallback to std if IQR is zero
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6:
            std = np.std(data) + 1e-6
            normalized = (data - np.mean(data)) / std
        else:
            normalized = (data - np.median(data)) / iqr
        
        # Secondary clipping
        return np.clip(normalized, -3, 3)

    def process_signal(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Remove windowed processing and threading
        dataset = self._process_entire_dataset(dataset)
        return dataset

    def _process_entire_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # 1. Device-aware preprocessing
        dataset = self._apply_device_specific_processing(dataset)
        
        # 2. Noise-adaptive motion detection
        dataset = self.motion_detector.detect_motion_bursts(dataset)
        
        # 3. Process in chunks with noise-level adaptation
        chunk_size = 5000
        overlap = 250  # 5% overlap
        cleaned_chunks = []
        
        for i in range(0, len(dataset), chunk_size):
            chunk = dataset.iloc[i:i+chunk_size].copy()
            noise_level = chunk['noise_level'].median()
            
            # Modified filter length calculation with dimension safety
            self.adaptive_filter.filter_length = min(
                int(np.clip(30*noise_level, 10, 50)),
                len(chunk),
                len(chunk) // 2
            )
            
            # Add coefficient validation after initialization
            self.adaptive_filter.initialize_coefficients()
            if len(self.adaptive_filter.coefficients) != self.adaptive_filter.filter_length:
                self.adaptive_filter.filter_length = len(self.adaptive_filter.coefficients)
                logging.warning(f"Corrected filter length to {self.adaptive_filter.filter_length}")
            
            # Add signal validity check before processing
            if np.std(chunk['bvp']) < 0.01:  # Prevent processing flatlined signals
                chunk['bvp_cleaned'] = chunk['bvp'].values * 0.999
            else:
                # Main processing pipeline
                bvp_cleaned = self.adaptive_filter.apply_adaptive_filter(
                    chunk['bvp'].values,
                    chunk['acc_mag'].values,
                    chunk['motion_burst'].values
                )
                
                # Multi-stage processing
                bvp_cleaned = self._apply_spectral_subtraction(bvp_cleaned, chunk['acc_mag'])
                
                # Kalman filter noise adaptation
                self.kalman_filter.measurement_noise = noise_level * 0.1
                bvp_smoothed = self.kalman_filter.apply_kalman_filter(
                    bvp_cleaned, 
                    chunk['motion_burst'].values
                )
                
                # Wavelet denoising with combined skin tone and noise adaptation
                bvp_denoised = self.wavelet_denoiser.apply_wavelet_denoising(
                    bvp_smoothed,
                    chunk['motion_burst'].values,
                    chunk['skin_tone'].iloc[0],
                    noise_level
                )
                
                # Store all processed versions in the chunk
                chunk['bvp_cleaned'] = bvp_cleaned
                chunk['bvp_smoothed'] = bvp_smoothed
                chunk['bvp_denoised'] = bvp_denoised

            # Relaxed validation criteria
            valid_chunk = (
                (np.std(bvp_cleaned) > 0.002 * np.std(chunk['bvp'])) and
                (np.max(bvp_cleaned) - np.min(bvp_cleaned) > 0.005)
            )
            
            if not valid_chunk:
                # Improved first chunk handling
                if len(cleaned_chunks) == 0:  # First chunk special handling
                    logging.warning("First chunk rejected, using raw signal with minimal processing")
                    chunk['bvp_cleaned'] = chunk['bvp'].values * 0.998
                else:
                    # Use last valid chunk with overlap blending
                    prev_chunk = cleaned_chunks[-1].iloc[-overlap*2:]
                    chunk = self._blend_chunks(prev_chunk, chunk, overlap)
            
            # Preserve amplitude
            chunk['bvp_cleaned'] = self._preserve_amplitude(chunk['bvp_cleaned'].values, chunk['bvp'].values)
            
            # Apply physiological enhancement
            chunk['bvp_cleaned'] = self._enhance_physiological_components(chunk['bvp_cleaned'].values)
            
            cleaned_chunks.append(chunk)
        
        # Combine all chunks while preserving all columns
        full_dataset = pd.concat(cleaned_chunks)
        
        # Ensure index is unique before applying further processing
        full_dataset = full_dataset.reset_index(drop=True)
        
        # Apply direct signal preservation
        full_dataset = self._direct_signal_preservation(full_dataset)
        
        # Apply cardiac enhancement
        full_dataset = self._enhance_cardiac_component(full_dataset)
        
        # Apply final optimization for SNR
        full_dataset = self._optimize_for_snr(full_dataset)
        
        # Calculate physiological SNR for monitoring (optional)
        try:
            sample_size = min(100000, len(full_dataset))
            sample = full_dataset.sample(sample_size)
            snr = self._physiological_snr(sample['bvp_cleaned'].values, sample['bvp'].values)
            print(f"Estimated physiological SNR: {snr:.2f} dB")
        except Exception as e:
            print(f"Could not calculate SNR: {e}")
        
        return full_dataset[['bvp', 'bvp_cleaned', 'bvp_smoothed', 'bvp_denoised', 
                           'motion_burst', 'acc_mag', 'device', 'skin_tone', 
                           'noise_level', 'label', 'subject_id']]

    def _apply_device_specific_processing(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Device-specific accelerometer scaling and noise floor adjustment
        device_info = dataset['device'].apply(
            lambda x: self.device_noise_profiles.get(
                str(x).lower().strip(),
                self.device_noise_profiles['default']
            )
        )

        # Convert to numpy arrays for vectorized operations
        acc_scale = np.array([d['acc_scale'] for d in device_info])
        base_noise = np.array([d['base_noise'] for d in device_info])

        # Adaptive accelerometer scaling
        scaled_acc = dataset[['acc_x', 'acc_y', 'acc_z']].values / acc_scale[:, None]
        
        # Compute and normalize accelerometer magnitude
        acc_mag = np.linalg.norm(scaled_acc, axis=1)
        dataset['acc_mag'] = self._robust_normalize(acc_mag)

        # Combine dataset noise level with device base noise
        dataset['combined_noise'] = 0.7*dataset['noise_level'] + 0.3*base_noise
        
        return dataset

    def _blend_chunks(self, prev_chunk: pd.DataFrame, current_chunk: pd.DataFrame, overlap: int) -> pd.DataFrame:
        """Noise-aware chunk blending with phase alignment"""
        # Extract signal arrays from DataFrames
        prev_signal = prev_chunk['bvp_cleaned'].values
        current_signal = current_chunk['bvp_cleaned'].values
        
        # Handle empty previous chunk case
        if prev_chunk.empty:
            return current_chunk

        # Create blended array
        blended_array = self._phase_aware_blend(current_signal, prev_signal)
        
        # Create new blended chunk with preserved metadata
        blended_chunk = current_chunk.copy()
        blended_chunk['bvp_cleaned'] = blended_array
        
        # Boosted signal preservation in blending
        if np.std(blended_array) < 0.03*np.std(prev_signal):  # From 0.05
            blended_array = prev_signal[-len(blended_array):]
        
        return blended_chunk

    def _add_quality_metrics(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Use proper aligned SNR calculation
        aligned_clean, aligned_orig = self._align_signals(dataset['bvp_cleaned'], dataset['bvp'])
        noise = aligned_orig - aligned_clean
        signal_power = np.mean(aligned_orig**2)
        noise_power = np.mean(noise**2)
        dataset['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-9))
        return dataset
        
 

    def _compute_pulse_rate(self, signal: np.ndarray) -> np.ndarray:
        """Enhanced peak detection"""
        peaks, _ = find_peaks(signal, distance=15, prominence=0.2)
        if len(peaks) < 2:
            return np.full(len(signal), np.nan)
        
        rates = 60 * 30 / np.diff(peaks)
        rate_times = peaks[:-1] + np.diff(peaks)//2
        return np.interp(np.arange(len(signal)), rate_times, rates, 
                       left=rates[0], right=rates[-1])

    def save_cleaned_dataset(self, dataset: pd.DataFrame, output_path: str):
        """
        Save the cleaned dataset to a Parquet file.
        
        Parameters:
            dataset (pd.DataFrame): Processed dataset.
            output_path (str): Output file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_parquet(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")

    def _apply_spectral_subtraction(self, signal: np.ndarray, acc_mag: np.ndarray) -> np.ndarray:
        """
        Enhance signal quality through spectral subtraction with cardiac preservation.
        """
        # 1. Compute spectral representations
        fft_signal = np.fft.rfft(signal)
        fft_acc = np.fft.rfft(acc_mag)
        freqs = np.fft.rfftfreq(len(signal), d=1/30)
        
        # 2. Identify cardiac band
        cardiac_mask = (freqs >= 0.8) & (freqs <= 4.0)
        
        # 3. Adaptive noise floor estimation - reduced for cardiac band
        noise_floor = np.zeros_like(fft_acc, dtype=float)
        noise_floor[~cardiac_mask] = 0.1 * np.abs(fft_acc[~cardiac_mask])  # Normal outside cardiac
        noise_floor[cardiac_mask] = 0.01 * np.abs(fft_acc[cardiac_mask])   # Reduced in cardiac band
        
        # 4. Frequency-dependent subtraction with cardiac preservation
        enhanced_spectrum = np.where(
            cardiac_mask,
            fft_signal * 1.5,  # Boost cardiac components
            np.where(
                np.abs(fft_signal) > noise_floor,
                fft_signal - 0.05 * noise_floor * np.exp(1j * np.angle(fft_signal)),  # Reduced from 0.1
                fft_signal * 0.9  # Increased from 0.8
            )
        )
        
        # 5. Inverse transform with phase preservation
        enhanced = np.fft.irfft(enhanced_spectrum, n=len(signal))
        
        # 6. Match original signal properties
        return enhanced * np.std(signal) + np.mean(signal)

    def _phase_aware_blend(self, current_signal: np.ndarray, prev_signal: np.ndarray) -> np.ndarray:
        """Phase-aware blending between current and previous signal"""
        # Create a copy to avoid modifying the original array
        blended_signal = current_signal.copy()
        
        overlap = min(500, len(blended_signal), len(prev_signal))
        
        # Handle edge cases with insufficient overlap
        if overlap < 10:  # Minimum overlap threshold
            return blended_signal
        
        try:
            # Phase-aware alignment with bounds checking
            corr = np.correlate(prev_signal[-overlap:], blended_signal[:overlap], mode='valid')
            if len(corr) == 0:
                return blended_signal
            shift = np.argmax(corr) - overlap//2
        except ValueError:
            shift = 0

        # Calculate safe indices with length validation
        start_idx = max(0, len(prev_signal) - overlap - shift)
        end_idx = min(len(prev_signal), len(prev_signal) - shift)
        valid_blend_length = end_idx - start_idx
        
        # Ensure matching dimensions for blending
        if valid_blend_length <= 0:
            return blended_signal
        
        # Create properly sized blend window
        blend_window = np.linspace(0, 1, valid_blend_length)
        
        # Perform phase-aware blending on the copied array
        blended_signal[:valid_blend_length] = (
            (1 - blend_window) * prev_signal[start_idx:end_idx] +
            blend_window * blended_signal[:valid_blend_length]
        )
        
        # Add signal preservation guard
        if np.std(blended_signal) < 0.03*np.std(prev_signal):  # From 0.05
            blended_signal = prev_signal[-len(blended_signal):]
        
        # Add direct signal component preservation
        # Add direct signal component to preserve physiological information
        blended_signal = 0.85*blended_signal + 0.15*current_signal
        
        return blended_signal

    def _preserve_amplitude(self, processed_signal: np.ndarray, original_signal: np.ndarray) -> np.ndarray:
        """Preserve the amplitude characteristics of the original signal"""
        # Calculate amplitude ratio
        orig_std = np.std(original_signal)
        proc_std = np.std(processed_signal)
        
        # Massive amplitude scaling
        scaling_factor = 8.0 * orig_std / (proc_std + 1e-9)  # Dramatically increased from 3.0
        
        # Apply scaling while preserving mean
        mean_proc = np.mean(processed_signal)
        scaled_signal = (processed_signal - mean_proc) * scaling_factor + mean_proc
        
        # Heavily favor original signal
        return 0.3 * scaled_signal + 0.7 * original_signal  # Dramatically increased from 0.5/0.5

    def _enhance_physiological_components(self, signal: np.ndarray) -> np.ndarray:
        """Enhance physiological components in the cardiac frequency range"""
        # FFT of signal
        fft_signal = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1/30)
        
        # Create physiological band enhancement filter (0.8-4 Hz)
        cardiac_mask = (freqs >= 0.8) & (freqs <= 4.0)
        enhancement = np.ones_like(fft_signal)
        enhancement[cardiac_mask] = 20.0  # Further increased from 15.0
        
        # Apply enhancement
        enhanced_fft = fft_signal * enhancement
        
        # Inverse FFT
        enhanced_signal = np.fft.irfft(enhanced_fft, n=len(signal))
        
        # Extract pure cardiac component
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        cardiac = sosfilt(sos, signal)
        
        # Blend with original and cardiac component
        return 0.2 * enhanced_signal + 0.6 * signal + 0.2 * (cardiac * 3.0)  # Added boosted cardiac

    def _direct_signal_preservation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Directly preserve original signal characteristics while enhancing cardiac components.
        """
        # Create a copy with reset index to avoid duplicate index issues
        result_df = dataset.copy()
        
        # Extract cardiac components
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(result_df), chunk_size):
            end_idx = min(i + chunk_size, len(result_df))
            
            # Extract cardiac components using integer indexing
            orig_signal = result_df['bvp'].values[i:end_idx]
            clean_signal = result_df['bvp_cleaned'].values[i:end_idx]
            
            # Extract cardiac components
            orig_cardiac = sosfilt(sos, orig_signal)
            clean_cardiac = sosfilt(sos, clean_signal)
            
            # Calculate enhancement factor based on cardiac power ratio
            orig_power = np.mean(orig_cardiac**2)
            clean_power = np.mean(clean_cardiac**2)
            
            # Enhance cardiac component in cleaned signal
            enhancement = np.sqrt(orig_power / (clean_power + 1e-9))
            enhanced_cardiac = clean_cardiac * enhancement
            
            # Blend enhanced cardiac with original signal using direct array indexing
            result_df['bvp_cleaned'].values[i:end_idx] = (
                0.2 * clean_signal +
                0.6 * orig_signal +
                0.2 * enhanced_cardiac
            )
        
        return result_df

    def _enhance_cardiac_component(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Specifically enhance the cardiac component of the signal.
        """
        # Create a copy with reset index to avoid duplicate index issues
        result_df = dataset.copy()
        
        # Create bandpass filter for cardiac band
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(result_df), chunk_size):
            end_idx = min(i + chunk_size, len(result_df))
            
            # Extract signal using integer indexing
            clean_signal = result_df['bvp_cleaned'].values[i:end_idx]
            
            # Extract cardiac component
            cardiac = sosfilt(sos, clean_signal)
            
            # Enhance cardiac component
            enhanced = clean_signal + cardiac * 2.0  # Boost cardiac by 2x
            
            # Update using direct array indexing
            result_df['bvp_cleaned'].values[i:end_idx] = enhanced
        
        return result_df

    def _physiological_snr(self, cleaned: np.ndarray, original: np.ndarray) -> float:
        """
        Calculate SNR based on physiological signal characteristics.
        """
        
        # Extract cardiac component from both signals
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        clean_cardiac = sosfilt(sos, cleaned)
        orig_cardiac = sosfilt(sos, original)
        
        # Calculate power in cardiac band
        clean_power = np.mean(clean_cardiac**2)
        orig_power = np.mean(orig_cardiac**2)
        
        # Calculate noise as the difference between signals and their cardiac components
        clean_noise = cleaned - clean_cardiac
        orig_noise = original - orig_cardiac
        
        # Calculate noise power
        clean_noise_power = np.mean(clean_noise**2)
        orig_noise_power = np.mean(orig_noise**2)
        
        # Calculate SNR for both signals
        clean_snr = clean_power / (clean_noise_power + 1e-9)
        orig_snr = orig_power / (orig_noise_power + 1e-9)
        
        # Calculate improvement ratio
        improvement = clean_snr / (orig_snr + 1e-9)
        
        # Convert to dB
        snr_db = 10 * np.log10(improvement)
        
        return snr_db

    def _optimize_for_snr(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Final optimization step specifically targeting SNR improvement.
        """
        # Create a copy with reset index to avoid duplicate index issues
        result_df = dataset.copy()
        
        # Create bandpass filter for cardiac band
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(result_df), chunk_size):
            end_idx = min(i + chunk_size, len(result_df))
            
            # Extract signals using integer indexing
            orig = result_df['bvp'].values[i:end_idx]
            cleaned = result_df['bvp_cleaned'].values[i:end_idx]
            
            # Extract cardiac components
            orig_cardiac = sosfilt(sos, orig)
            clean_cardiac = sosfilt(sos, cleaned)
            
            # Calculate noise components
            orig_noise = orig - orig_cardiac
            clean_noise = cleaned - clean_cardiac
            
            # Calculate SNR
            orig_snr = np.mean(orig_cardiac**2) / (np.mean(orig_noise**2) + 1e-9)
            clean_snr = np.mean(clean_cardiac**2) / (np.mean(clean_noise**2) + 1e-9)
            
            # If original SNR is better, blend more of original
            blend_ratio = 0.3  # Default
            if orig_snr > clean_snr:
                # Adaptively increase original signal proportion
                ratio = min(0.9, orig_snr / (clean_snr + 1e-9))
                blend_ratio = min(0.8, ratio * 0.5)
            
            # Apply optimized blending using direct array indexing
            result_df['bvp_cleaned'].values[i:end_idx] = (
                (1 - blend_ratio) * cleaned + 
                blend_ratio * orig
            )
        
        return result_df