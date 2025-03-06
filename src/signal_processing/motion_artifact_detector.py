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
        # Remove G-force conversion (data already in Gs)
        # dataset['acc_x'] = ...  # Comment out conversion
        
        # Compute magnitude with safety checks
        dataset['acc_mag'] = np.sqrt(
            dataset['acc_x']**2 + 
            dataset['acc_y']**2 + 
            dataset['acc_z']**2
        ).clip(upper=20)  # Cap unrealistic values
        
        # Dynamic threshold based on moving percentile
        window_size = int(self.sampling_rate * 5)  # 5-second window
        dataset['acc_threshold'] = dataset['acc_mag'].rolling(
            window=window_size,
            min_periods=1,
            center=True
        ).quantile(0.9)  # 90th percentile
        
        # Motion score calculation
        motion_score = (dataset['acc_mag'] - dataset['acc_threshold']).clip(lower=0)
        max_score = motion_score.rolling(window=window_size).max()
        dataset['motion_burst'] = (max_score / (max_score.max() + 1e-9)).fillna(0)
        
        return dataset