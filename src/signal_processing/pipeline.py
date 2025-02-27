import pandas as pd
import numpy as np
import os
from .adaptive_filter import AdaptiveFilter
from .kalman_filter import KalmanFilter
from .wavelet_denoiser import WaveletDenoiser
from .motion_artifact_detector import MotionArtifactDetector

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
        self.wavelet_denoiser = WaveletDenoiser(wavelet='db4', level=3)
        self.motion_artifact_detector = MotionArtifactDetector()

    def process_signal(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Process the PPG signal using motion artifact detection, adaptive filtering, Kalman filtering, and wavelet denoising.
        
        Parameters:
            dataset (pd.DataFrame): Input dataset with 'bvp' and 'acc_mag' columns.
        
        Returns:
            pd.DataFrame: Dataset with cleaned 'bvp_cleaned' column.
        """
        # Compute accelerometer magnitude
        dataset['acc_mag'] = np.sqrt(dataset['acc_x']**2 + dataset['acc_y']**2 + dataset['acc_z']**2)

        # Robust normalization (use IQR)
        acc_median = dataset['acc_mag'].median()
        acc_iqr = dataset['acc_mag'].quantile(0.75) - dataset['acc_mag'].quantile(0.25)
        dataset['acc_mag'] = (dataset['acc_mag'] - acc_median) / (acc_iqr + 1e-6)

        # Detect motion artifacts
        dataset = self.motion_artifact_detector.detect_motion_bursts(dataset)

        # Apply LMS adaptive filtering
        bvp_cleaned = self.adaptive_filter.apply_adaptive_filter(
            noisy_signal=dataset['bvp'].values,
            reference_signal=dataset['acc_mag'].values,
            motion_burst=dataset['motion_burst'].values
        )
        if np.any(np.isnan(bvp_cleaned)):
            raise ValueError("Adaptive filtering produced NaNs")
        
        if np.all(bvp_cleaned == 0):
            raise ValueError("Adaptive filter output is entirely zero. Check accelerometer normalization.")

        # Apply Kalman filtering
        bvp_smoothed = self.kalman_filter.apply_kalman_filter(bvp_cleaned, dataset['motion_burst'].values)
        if np.any(np.isnan(bvp_smoothed)):
            raise ValueError("Kalman filtering produced NaNs")

        # Apply wavelet denoising
        bvp_denoised = self.wavelet_denoiser.apply_wavelet_denoising(bvp_smoothed, dataset['motion_burst'].values)
        if np.any(np.isnan(bvp_denoised)):
            raise ValueError("Wavelet denoising produced NaNs")

        # Add cleaned signal to dataset
        dataset['bvp_cleaned'] = bvp_denoised

        return dataset

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
