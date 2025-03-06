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

class SignalProcessingPipeline:
    def __init__(self, learning_rate: float = 0.01, filter_length: int = 10, process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        """
        Initialize the signal processing pipeline.
        
        Parameters:
            learning_rate (float): Learning rate for LMS adaptive filter.
            filter_length (int): Length of the LMS filter.
            process_noise (float): Process noise for Kalman filter.
            measurement_noise (float): Measurement noise for Kalman filter.
        """
        self.adaptive_filter = AdaptiveFilter(learning_rate=learning_rate, filter_length=filter_length)
        self.kalman_filter = KalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        self.wavelet_denoiser = WaveletDenoiser()
        self.motion_artifact_detector = MotionArtifactDetector()

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
        """
        Process the PPG signal using motion artifact detection, adaptive filtering, Kalman filtering, and wavelet denoising.
        
        Parameters:
            dataset (pd.DataFrame): Input dataset with 'bvp' and 'acc_mag' columns.
        
        Returns:
            pd.DataFrame: Dataset with cleaned 'bvp_cleaned' column.
        """
        # Precompute motion bursts once
        motion_bursts = self.motion_artifact_detector.detect_motion_bursts(dataset)
        
        # Split into 5-second windows (150 samples @30Hz)
        window_size = 150
        windows = [dataset.iloc[i:i+window_size] for i in range(0, len(dataset), window_size)]
        
        # Process windows in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            with tqdm(total=len(windows)) as pbar:
                results = list(executor.map(self._process_window, windows))
                for _ in results:
                    pbar.update()
        
        return pd.concat(results)

    def _process_window(self, window: pd.DataFrame) -> pd.DataFrame:
        # Reset index to ensure unique row identifiers
        window = window.reset_index(drop=True)

        # Compute and normalize accelerometer magnitude
        window['acc_mag'] = np.sqrt(window['acc_x']**2 + window['acc_y']**2 + window['acc_z']**2)
        window['acc_mag'] = self._robust_normalize(window['acc_mag'].values)
        
        # Detect motion artifacts
        window = self.motion_artifact_detector.detect_motion_bursts(window)

        # Apply LMS adaptive filtering
        bvp_cleaned = self.adaptive_filter.apply_adaptive_filter(
            noisy_signal=window['bvp'].values,
            reference_signal=window['acc_mag'].values,
            motion_burst=window['motion_burst'].values
        )
        if np.any(np.isnan(bvp_cleaned)):
            raise ValueError("Adaptive filtering produced NaNs")
        
        if np.all(bvp_cleaned == 0):
            raise ValueError("Adaptive filter output is entirely zero. Check accelerometer normalization.")

        # Apply Kalman filtering
        bvp_smoothed = self.kalman_filter.apply_kalman_filter(bvp_cleaned, window['motion_burst'].values)
        if np.any(np.isnan(bvp_smoothed)):
            raise ValueError("Kalman filtering produced NaNs")

        # Apply wavelet denoising with skin-tone adaptation
        bvp_denoised = self.wavelet_denoiser.apply_wavelet_denoising(
            signal=bvp_smoothed,
            motion_burst=window['motion_burst'].values,
            skin_tone=window['skin_tone'].iloc[0]
        )
        if np.any(np.isnan(bvp_denoised)):
            raise ValueError("Wavelet denoising produced NaNs")

        # Add cleaned signal to dataset
        window['bvp_cleaned'] = bvp_denoised

        # Add quality metrics
        window = self._add_quality_metrics(window)
        
        return window

    def _add_quality_metrics(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """SNR calculation using original signal as reference"""
        # Extract and validate signals
        original = dataset['bvp'].values.astype(np.float64)
        cleaned = dataset['bvp_cleaned'].values.astype(np.float64)
        
        # 1. Remove NaNs and ensure finite values
        original = original[np.isfinite(original)]
        cleaned = cleaned[np.isfinite(cleaned)]
        
        # 2. Match lengths using zero-padding
        max_len = max(len(original), len(cleaned))
        original = np.pad(original, (0, max_len - len(original)))
        cleaned = np.pad(cleaned, (0, max_len - len(cleaned)))
        
        # 3. Add DTW dimension validation
        if original.ndim != 1 or cleaned.ndim != 1:
            raise ValueError(f"Signal dimensions invalid: {original.shape} vs {cleaned.shape}")
        
        # 4. Use constrained DTW with length check
        if len(original) < 10 or len(cleaned) < 10:
            dataset['snr'] = np.nan
            return dataset
        
        # Replace with downsampled DTW
        downsample_factor = max(1, len(original) // 1000)  # Keep ~1000 points
        original_ds = original[::downsample_factor]
        cleaned_ds = cleaned[::downsample_factor]
        
        # Use constrained DTW
        distance, path = fastdtw(
            original_ds, 
            cleaned_ds,
            radius=5,
            dist=lambda x,y: abs(x-y)  # Manhattan faster than Euclidean
        )
        
        # Calculate SNR on aligned samples
        aligned_original = original[np.array([i for i, _ in path])]
        aligned_cleaned = cleaned[np.array([j for _, j in path])]
        
        signal_power = np.mean(aligned_original**2)
        noise_power = np.mean((aligned_cleaned - aligned_original)**2)
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