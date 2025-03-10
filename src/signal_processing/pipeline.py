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
        cleaned_chunks = []
        
        for i in range(0, len(dataset), chunk_size):
            chunk = dataset.iloc[i:i+chunk_size].copy()
            noise_level = chunk['noise_level'].median()
            
            # Adaptive filter tuning based on noise level
            self.adaptive_filter.learning_rate = np.clip(0.1/noise_level, 0.01, 0.5)
            filter_length = int(np.clip(30*noise_level, 10, 50))
            
            # Ensure filter length doesn't exceed chunk size
            self.adaptive_filter.filter_length = min(filter_length, len(chunk))
            
            # Initialize coefficients for each chunk
            self.adaptive_filter.initialize_coefficients()
            
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
            
            # Improved chunk blending
            chunk['bvp_cleaned'] = self._blend_chunks(
                bvp_denoised, 
                cleaned_chunks,
                noise_level=noise_level
            )
            cleaned_chunks.append(chunk)
        
        return pd.concat(cleaned_chunks)

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

    def _blend_chunks(self, current_signal: np.ndarray, prev_chunks: list, noise_level: float) -> np.ndarray:
        """Noise-aware chunk blending with phase alignment"""
        if not prev_chunks:
            return current_signal
            
        prev_signal = prev_chunks[-1]['bvp_cleaned'].values
        overlap = min(500, len(current_signal), len(prev_signal))
        
        # Handle edge cases with insufficient overlap
        if overlap < 10:  # Minimum overlap threshold
            return current_signal
        
        try:
            # Phase-aware alignment with bounds checking
            corr = np.correlate(prev_signal[-overlap:], current_signal[:overlap], mode='valid')
            if len(corr) == 0:
                return current_signal
            shift = np.argmax(corr) - overlap//2
        except ValueError:
            shift = 0

        # Calculate safe indices with length validation
        start_idx = max(0, len(prev_signal) - overlap - shift)
        end_idx = min(len(prev_signal), len(prev_signal) - shift)  # Clamp to signal length
        valid_blend_length = end_idx - start_idx
        
        # Ensure matching dimensions for blending
        if valid_blend_length <= 0:
            return current_signal
        
        # Create properly sized blend window
        blend_window = np.linspace(0, 1, valid_blend_length)
        
        # Perform dimensionally safe blending
        current_signal[:valid_blend_length] = (
            (1 - blend_window) * prev_signal[start_idx:end_idx] +
            blend_window * current_signal[:valid_blend_length]
        )
        
        return current_signal

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
        freqs = np.fft.rfftfreq(len(signal))
        
        # 2. Adaptive noise floor estimation
        noise_floor = 0.7 * np.abs(fft_acc) * (1 + np.linspace(0, 1, len(fft_acc)))
        
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