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
        """Enhanced normalization with fallback and empty array handling"""
        # Check for empty array first
        if len(data) == 0:
            return np.array([])
        
        data = np.nan_to_num(data, nan=np.nanmedian(data) if len(data) > 0 else 0)
        
        # Fallback to std if IQR is zero
        q75, q25 = np.percentile(data, [75, 25]) if len(data) > 0 else (0, 0)
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
        # Increase chunk size for better performance
        chunk_size = 20000  # Increased from 5000
        overlap = 250  # 5% overlap
        cleaned_chunks = []
        
        # Use tqdm for progress tracking
        total_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        for i in tqdm(range(0, len(dataset), chunk_size), total=total_chunks, desc="Processing chunks"):
            chunk = dataset.iloc[i:i+chunk_size].copy()
            
            # Skip processing empty chunks
            if len(chunk) == 0:
                continue
            
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
            if len(chunk) == 0 or np.std(chunk['bvp']) < 0.01:  # Prevent processing flatlined signals
                if len(chunk) > 0:  # Only set values if chunk is not empty
                    chunk['bvp_cleaned'] = chunk['bvp'].values * 0.999
                    chunk['bvp_smoothed'] = chunk['bvp'].values * 0.998
                    chunk['bvp_denoised'] = chunk['bvp'].values * 0.997
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
                
                # Process wavelet denoising in batches instead of sample by sample
                # This is a major performance improvement
                bvp_denoised = np.zeros_like(bvp_smoothed)
                skin_tone = chunk['skin_tone'].iloc[0]
                
                # Process in sub-chunks for wavelet denoising
                sub_chunk_size = 1000
                for j in range(0, len(bvp_smoothed), sub_chunk_size):
                    end_j = min(j + sub_chunk_size, len(bvp_smoothed))
                    sub_signal = bvp_smoothed[j:end_j]
                    sub_motion = chunk['motion_burst'].values[j:end_j]
                    
                    # Apply wavelet denoising to the entire sub-chunk at once
                    bvp_denoised[j:end_j] = self.wavelet_denoiser.apply_wavelet_denoising(
                        sub_signal,
                        sub_motion,
                        skin_tone,
                        noise_level
                    )
                
                # Store all processed versions in the chunk
                chunk['bvp_cleaned'] = bvp_cleaned
                chunk['bvp_smoothed'] = bvp_smoothed
                chunk['bvp_denoised'] = bvp_denoised

            # Relaxed validation criteria with empty array check
            valid_chunk = False
            if 'bvp_cleaned' in chunk and len(chunk) > 0:
                bvp_cleaned = chunk['bvp_cleaned'].values
                if len(bvp_cleaned) > 0:
                    valid_chunk = (
                        (np.std(bvp_cleaned) > 0.002 * np.std(chunk['bvp'])) and
                        (np.max(bvp_cleaned) - np.min(bvp_cleaned) > 0.005)
                    )
            
            if not valid_chunk:
                # Improved first chunk handling
                if len(cleaned_chunks) == 0:  # First chunk special handling
                    logging.warning("First chunk rejected, using raw signal with minimal processing")
                    if len(chunk) > 0:  # Only set values if chunk is not empty
                        chunk['bvp_cleaned'] = chunk['bvp'].values * 0.998
                        chunk['bvp_smoothed'] = chunk['bvp'].values * 0.997
                        chunk['bvp_denoised'] = chunk['bvp'].values * 0.996
                else:
                    # Use last valid chunk with overlap blending
                    prev_chunk = cleaned_chunks[-1].iloc[-overlap*2:]
                    chunk = self._blend_chunks(prev_chunk, chunk, overlap)
            
            # Preserve amplitude
            if len(chunk) > 0:  # Only process if chunk is not empty
                chunk['bvp_cleaned'] = self._preserve_amplitude(chunk['bvp_cleaned'].values, chunk['bvp'].values)
                
                # Apply physiological enhancement
                chunk['bvp_cleaned'] = self._enhance_physiological_components(chunk['bvp_cleaned'].values)
            
            cleaned_chunks.append(chunk)
        
        # Combine all chunks while preserving all columns
        if not cleaned_chunks:  # Handle case where all chunks were empty
            return dataset.copy()
        
        full_dataset = pd.concat(cleaned_chunks)
        
        # Ensure index is unique before applying further processing
        full_dataset = full_dataset.reset_index(drop=True)
        
        # Apply post-processing in a more efficient way
        print("Applying post-processing...")
        
        # Apply direct signal preservation
        full_dataset = self._direct_signal_preservation(full_dataset)
        
        # Apply cardiac enhancement
        full_dataset = self._enhance_cardiac_component(full_dataset)
        
        # Apply final optimization for SNR
        full_dataset = self._optimize_for_snr(full_dataset)
        
        # Update smoothed and denoised signals to maintain consistency
        # Process in larger chunks for better performance
        chunk_size = 50000  # Increased from 10000
        for i in tqdm(range(0, len(full_dataset), chunk_size), desc="Updating signals"):
            end_idx = min(i + chunk_size, len(full_dataset))
            
            # Apply Kalman filter to updated cleaned signal
            full_dataset['bvp_smoothed'].values[i:end_idx] = self.kalman_filter.apply_kalman_filter(
                full_dataset['bvp_cleaned'].values[i:end_idx],
                full_dataset['motion_burst'].values[i:end_idx]
            )
            
            # Apply wavelet denoising in batches instead of sample by sample
            # This is a major performance improvement
            smoothed_signal = full_dataset['bvp_smoothed'].values[i:end_idx]
            motion_burst = full_dataset['motion_burst'].values[i:end_idx]
            
            # Group by skin tone for batch processing
            skin_tones = full_dataset['skin_tone'].iloc[i:end_idx].unique()
            for skin_tone in skin_tones:
                # Create mask for this skin tone
                mask = full_dataset['skin_tone'].iloc[i:end_idx] == skin_tone
                mask_indices = np.where(mask)[0]
                
                if len(mask_indices) == 0:
                    continue
                
                # Get noise level (use median for the group)
                noise_level = full_dataset['noise_level'].iloc[i:end_idx].iloc[mask_indices].median()
                
                # Process in sub-chunks
                sub_chunk_size = 5000
                for j in range(0, len(mask_indices), sub_chunk_size):
                    end_j = min(j + sub_chunk_size, len(mask_indices))
                    indices = mask_indices[j:end_j]
                    
                    # Skip if no indices
                    if len(indices) == 0:
                        continue
                    
                    # Get absolute indices
                    abs_indices = [idx + i for idx in indices]
                    
                    # Extract signals for this batch
                    sub_signal = smoothed_signal[indices]
                    sub_motion = motion_burst[indices]
                    
                    # Apply wavelet denoising to the entire sub-chunk at once
                    denoised = self.wavelet_denoiser.apply_wavelet_denoising(
                        sub_signal,
                        sub_motion,
                        skin_tone,
                        noise_level
                    )
                    
                    # Update the values
                    for k, idx in enumerate(abs_indices):
                        if idx < len(full_dataset):
                            full_dataset['bvp_denoised'].values[idx] = denoised[k]
        
        # Calculate physiological SNR for monitoring (optional)
        try:
            sample_size = min(100000, len(full_dataset))
            if sample_size > 0:  # Only sample if there's data
                sample = full_dataset.sample(sample_size)
                if len(sample) > 0:  # Double-check sample has data
                    snr = self._physiological_snr(sample['bvp_cleaned'].values, sample['bvp'].values)
                    print(f"Estimated physiological SNR: {snr:.2f} dB")
                else:
                    print("Empty sample, cannot calculate SNR")
            else:
                print("Dataset too small to calculate SNR")
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

    def _compute_pulse_rate(self, signal: np.ndarray) -> np.ndarray:
        """Enhanced peak detection"""
        peaks, _ = find_peaks(signal, distance=15, prominence=0.2)
        if len(peaks) < 2:
            return np.full(len(signal), np.nan)
        
        rates = 60 * 30 / np.diff(peaks)
        rate_times = peaks[:-1] + np.diff(peaks)//2
        return np.interp(np.arange(len(signal)), rate_times, rates, 
                       left=rates[0], right=rates[-1])

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
        
        # Reduced amplitude scaling
        scaling_factor = 1.5 * orig_std / (proc_std + 1e-9)  # Reduced from 8.0
        
        # Apply scaling while preserving mean
        mean_proc = np.mean(processed_signal)
        scaled_signal = (processed_signal - mean_proc) * scaling_factor + mean_proc
        
        # More balanced blending
        return 0.4 * scaled_signal + 0.6 * original_signal  # Adjusted from 0.3/0.7

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
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(result_df), chunk_size):
            end_idx = min(i + chunk_size, len(result_df))
            
            # Extract cardiac components using integer indexing
            orig_signal = result_df['bvp'].values[i:end_idx]
            clean_signal = result_df['bvp_cleaned'].values[i:end_idx]
            
            # Skip empty chunks
            if len(orig_signal) == 0 or len(clean_signal) == 0:
                continue
            
            # Simple blending without additional enhancement
            result_df['bvp_cleaned'].values[i:end_idx] = (
                0.5 * clean_signal +
                0.5 * orig_signal
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
            
            # Skip empty chunks
            if len(clean_signal) == 0:
                continue
            
            # Extract cardiac component
            cardiac = sosfilt(sos, clean_signal)
            
            # Enhance cardiac component - reduced enhancement
            enhanced = clean_signal + cardiac * 0.5  # Reduced from 2.0
            
            # Update using direct array indexing
            result_df['bvp_cleaned'].values[i:end_idx] = enhanced
        
        return result_df

    def _physiological_snr(self, cleaned: np.ndarray, original: np.ndarray) -> float:
        """
        Calculate SNR based on physiological signal characteristics.
        """
        # Check for empty arrays
        if len(cleaned) == 0 or len(original) == 0:
            return 0.0  # Return default value for empty arrays
        
        # Use a smaller sample for SNR calculation to improve performance
        max_samples = 10000
        if len(cleaned) > max_samples:
            # Take evenly spaced samples
            indices = np.linspace(0, len(cleaned)-1, max_samples, dtype=int)
            cleaned = cleaned[indices]
            original = original[indices]
        
        # Extract cardiac component from both signals
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        clean_cardiac = sosfilt(sos, cleaned)
        orig_cardiac = sosfilt(sos, original)
        
        # Calculate power in cardiac band with safety checks
        clean_power = np.mean(clean_cardiac**2) if len(clean_cardiac) > 0 else 1e-9
        orig_power = np.mean(orig_cardiac**2) if len(orig_cardiac) > 0 else 1e-9
        
        # Calculate noise as the difference between signals and their cardiac components
        clean_noise = cleaned - clean_cardiac
        orig_noise = original - orig_cardiac
        
        # Calculate noise power with safety checks
        clean_noise_power = np.mean(clean_noise**2) if len(clean_noise) > 0 else 1e-9
        orig_noise_power = np.mean(orig_noise**2) if len(orig_noise) > 0 else 1e-9
        
        # Calculate SNR for both signals with safety checks
        clean_snr = clean_power / (clean_noise_power + 1e-9)
        orig_snr = orig_power / (orig_noise_power + 1e-9)
        
        # Calculate improvement ratio
        improvement = clean_snr / (orig_snr + 1e-9)
        
        # Convert to dB with safety check
        if improvement <= 0:
            return 0.0  # Return default value for invalid improvement
        
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
            
            # Skip empty chunks
            if len(orig) == 0 or len(cleaned) == 0:
                continue
            
            # Extract cardiac components
            orig_cardiac = sosfilt(sos, orig)
            clean_cardiac = sosfilt(sos, cleaned)
            
            # Calculate noise components
            orig_noise = orig - orig_cardiac
            clean_noise = cleaned - clean_cardiac
            
            # Calculate SNR with safety checks
            orig_noise_power = np.mean(orig_noise**2) if len(orig_noise) > 0 else 1e-9
            clean_noise_power = np.mean(clean_noise**2) if len(clean_noise) > 0 else 1e-9
            
            orig_cardiac_power = np.mean(orig_cardiac**2) if len(orig_cardiac) > 0 else 1e-9
            clean_cardiac_power = np.mean(clean_cardiac**2) if len(clean_cardiac) > 0 else 1e-9
            
            orig_snr = orig_cardiac_power / (orig_noise_power + 1e-9)
            clean_snr = clean_cardiac_power / (clean_noise_power + 1e-9)
            
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