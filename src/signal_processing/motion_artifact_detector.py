# src/signal_processing/motion_artifact_detector.py

import numpy as np
import pandas as pd
import logging
from numpy.lib.stride_tricks import sliding_window_view

class MotionArtifactDetector:
    def __init__(self, acc_threshold_factor: float = 1.3,
                 burst_duration: float = 1.5, 
                 sampling_rate: int = 30):
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
        self.window_size = int(sampling_rate * 1.5)  # 1.5 second window

    def detect_motion_bursts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Device-aware accelerometer normalization
        device_scales = {
            'apple_watch': 2048,
            'galaxy_watch': 1024
        }
        scales = dataset['device'].str.lower().map(device_scales).fillna(512).values
        acc = dataset[['acc_x', 'acc_y', 'acc_z']].values / scales[:, None]
        
        # Noise-adaptive magnitude calculation
        acc_mag = np.linalg.norm(acc, axis=1)
        iqr = np.percentile(acc_mag, 75) - np.percentile(acc_mag, 25)
        dataset['acc_mag'] = (acc_mag - np.median(acc_mag)) / (iqr + 1e-9)
        
        # Improved noise-adaptive threshold
        dynamic_threshold = (
            np.median(acc_mag) + 
            1.2*np.std(acc_mag) * (1 + dataset['noise_level'].values)
        )
        
        # State machine with persistence
        motion_state = np.zeros(len(dataset))
        in_burst = False
        for i in range(1, len(dataset)):
            # Use .iloc for position-based indexing
            current_noise = dataset['noise_level'].iloc[i]
            current_threshold = dynamic_threshold[i]
            
            if acc_mag[i] > current_threshold * (1 + 0.3*current_noise):
                motion_state[i] = min(motion_state[i-1] + 0.2, 1.0)
                in_burst = True
            elif in_burst and (acc_mag[i] > 0.7*current_threshold):
                motion_state[i] = max(motion_state[i-1] - 0.05, 0.4)
            else:
                motion_state[i] = max(motion_state[i-1] - 0.1, 0.0)
                in_burst = False
        
        dataset['motion_burst'] = np.round(motion_state, 2)
        return dataset