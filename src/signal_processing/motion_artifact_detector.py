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
        
        # 1. Add G-force conversion (critical fix)
        dataset['acc_x'] = dataset['acc_x'] / 16384  # ±8g range
        dataset['acc_y'] = dataset['acc_y'] / 16384
        dataset['acc_z'] = dataset['acc_z'] / 16384
        
        # 2. Hybrid approach combining both methods
        dataset['acc_mag'] = np.sqrt(dataset['acc_x']**2 + dataset['acc_y']**2 + dataset['acc_z']**2)
        
        # Dynamic threshold (5s window 95th percentile)
        window_size = int(self.sampling_rate * 5)
        rolling_q95 = dataset['acc_mag'].rolling(window=window_size, min_periods=1, center=True).quantile(0.95)
        
        # Motion score with hysteresis
        motion_score = (dataset['acc_mag'] > rolling_q95 * 1.2).astype(float)
        dataset['motion_burst'] = motion_score.rolling(window=window_size, min_periods=1).mean()
        
        return dataset