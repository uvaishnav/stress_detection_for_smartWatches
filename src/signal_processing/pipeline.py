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
        self.adaptive_filter_state = None
        self.kalman_filter_state = None

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
        # Add chunked processing
        chunk_size = 10000  # Process 10k samples at a time
        total_samples = len(dataset)
        
        # Pre-process motion detection first
        dataset = self.motion_artifact_detector.detect_motion_bursts(dataset)
        
        # Process in chunks
        cleaned_chunks = []
        for i in range(0, total_samples, chunk_size):
            # Create explicit copy of the chunk
            chunk = dataset.iloc[i:i+chunk_size].copy()
            
            # Fix: Check if state exists using 'is not None'
            if self.adaptive_filter_state is not None:
                self.adaptive_filter.coefficients = self.adaptive_filter_state
            if self.kalman_filter_state is not None:
                self.kalman_filter.__dict__.update(self.kalman_filter_state)
            
            # Adaptive Filter
            bvp_cleaned = self.adaptive_filter.apply_adaptive_filter(
                chunk['bvp'].values,
                chunk['acc_mag'].values,
                chunk['motion_burst'].values
            )
            
            # Kalman Filter
            bvp_smoothed = self.kalman_filter.apply_kalman_filter(bvp_cleaned, chunk['motion_burst'].values)
            
            # Wavelet Denoising
            bvp_denoised = self.wavelet_denoiser.apply_wavelet_denoising(
                bvp_smoothed,
                chunk['motion_burst'].values,
                chunk['skin_tone'].iloc[0]
            )
            
            # Modify the copy directly
            chunk.loc[:, 'bvp_cleaned'] = bvp_denoised
            cleaned_chunks.append(chunk)
            
            # Save states
            self.adaptive_filter_state = self.adaptive_filter.coefficients.copy()
            self.kalman_filter_state = self.kalman_filter.__dict__.copy()
        
        return pd.concat(cleaned_chunks)

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