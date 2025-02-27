# src/signal_processing/motion_artifact_detector.py

import numpy as np
import pandas as pd

class MotionArtifactDetector:
    def __init__(self, acc_threshold_factor: float = 1.5, burst_duration: float = 1.5, sampling_rate: int = 30):
        """
        Initialize the motion artifact detector with dynamic thresholding.
        
        Parameters:
            acc_threshold_factor (float): Factor to scale the median accelerometer magnitude.
            burst_duration (float): Duration of motion bursts in seconds.
            sampling_rate (int): Sampling rate of the dataset in Hz.
        """
        self.acc_threshold_factor = acc_threshold_factor
        self.burst_duration = burst_duration
        self.sampling_rate = sampling_rate

    def detect_motion_bursts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Detect motion bursts in the dataset based on accelerometer data.
        
        Parameters:
            dataset (pd.DataFrame): Input dataset containing 'acc_x', 'acc_y', 'acc_z'.
        
        Returns:
            pd.DataFrame: Dataset with an additional 'motion_burst' column.
        """
        # Compute accelerometer magnitude
        dataset['acc_mag'] = np.sqrt(dataset['acc_x']**2 + dataset['acc_y']**2 + dataset['acc_z']**2)

        # Compute dynamic threshold based on median magnitude
        median_acc_mag = dataset['acc_mag'].median()
        if np.isnan(median_acc_mag) or median_acc_mag == 0:
            median_acc_mag = 1.0  # Default small value to avoid division by zero
        acc_threshold = median_acc_mag * self.acc_threshold_factor

        # Identify motion bursts
        window_size = int(self.burst_duration * self.sampling_rate)
        rolling_max = dataset['acc_mag'].rolling(window=window_size, center=True).max()
        dataset['motion_burst'] = (rolling_max > acc_threshold).astype(int)

        # Ensure no NaNs in motion_burst
        dataset['motion_burst'].fillna(0, inplace=True)

        return dataset