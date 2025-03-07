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
        
        # More realistic threshold (0.2g for subtle motions)
        acc_threshold = 0.2  # Reduced from 1.5g
        window_size = int(self.sampling_rate * 1)  # 1-second window
        
        dataset['acc_mag'] = np.sqrt(dataset['acc_x']**2 + dataset['acc_y']**2 + dataset['acc_z']**2)
        
        # Use efficient rolling maximum
        rolling_max = dataset['acc_mag'].rolling(window=window_size, min_periods=1, center=True).max()
        
        # Dynamic threshold based on baseline
        baseline = dataset['acc_mag'].quantile(0.95)  # 95th percentile as baseline
        dataset['motion_burst'] = (rolling_max > (baseline * 1.5)).astype(float)
        
        return dataset