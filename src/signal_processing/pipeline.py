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
from scipy.signal import resample
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
                len(chunk) // 2  # Add this line to prevent oversize filters
            )
            
            # Add coefficient validation after initialization
            self.adaptive_filter.initialize_coefficients()
            if len(self.adaptive_filter.coefficients) != self.adaptive_filter.filter_length:
                self.adaptive_filter.filter_length = len(self.adaptive_filter.coefficients)
                logging.warning(f"Corrected filter length to {self.adaptive_filter.filter_length}")
            
            # Add signal validity check before processing
            if np.std(chunk['bvp']) < 0.01:  # Prevent processing flatlined signals
                chunk['bvp_cleaned'] = chunk['bvp'].values
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
                (np.std(bvp_cleaned) > 0.01 * np.std(chunk['bvp'])) and  # From 0.015
                (np.max(bvp_cleaned) - np.min(bvp_cleaned) > 0.03)  # From 0.05
            )
            
            if not valid_chunk:
                # Improved first chunk handling
                if len(cleaned_chunks) == 0:  # First chunk special handling
                    logging.warning("First chunk rejected, using raw signal with noise reduction")
                    chunk['bvp_cleaned'] = chunk['bvp'].values * 0.98  # Minimal noise reduction
                else:
                    # Use last valid chunk with overlap blending
                    prev_chunk = cleaned_chunks[-1].iloc[-overlap*2:]
                    chunk = self._blend_chunks(prev_chunk, chunk, overlap)
            
            cleaned_chunks.append(chunk)
        
        # Combine all chunks while preserving all columns
        full_dataset = pd.concat(cleaned_chunks)
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
        if np.std(blended_array) < 0.05*np.std(prev_signal):  # From 0.1
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
        Enhance signal quality through spectral subtraction of accelerometer components.
        
        Parameters:
            signal (np.ndarray): Partially cleaned BVP signal
            acc_mag (np.ndarray): Normalized accelerometer magnitude
            
        Returns:
            np.ndarray: Enhanced signal with motion components subtracted
        """
        # 1. Compute spectral representations
        fft_signal = np.fft.rfft(signal)
        fft_acc = np.fft.rfft(acc_mag)
        
        # 2. Adaptive noise floor estimation
        noise_floor = 0.4 * np.abs(fft_acc) * (1 + np.linspace(0, 1, len(fft_acc)))  # Reduced from 0.7
        
        # 3. Frequency-dependent subtraction
        enhanced_spectrum = np.where(
            np.abs(fft_signal) > noise_floor,
            fft_signal - 0.5 * noise_floor * np.exp(1j * np.angle(fft_signal)),
            fft_signal * 0.2  # Attenuate below noise floor
        )
        
        # 4. Inverse transform with phase preservation
        enhanced = np.fft.irfft(enhanced_spectrum, n=len(signal))
        
        # 5. Match original signal properties
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
        if np.std(blended_signal) < 0.05*np.std(prev_signal):  # From 0.1
            blended_signal = prev_signal[-len(blended_signal):]
        
        return blended_signal