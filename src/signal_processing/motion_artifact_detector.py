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
        
        # Revised threshold calculation
        dynamic_threshold = (
            np.median(acc_mag) + 
            1.4*np.std(acc_mag) * (1 + 0.25*dataset['noise_level'].values)  # From 1.3
        )
        
        # State machine with persistence
        motion_state = np.zeros(len(dataset))
        in_burst = False
        for i in range(1, len(dataset)):
            # Use .iloc for position-based indexing
            current_noise = dataset['noise_level'].iloc[i]
            current_threshold = dynamic_threshold[i]
            
            # Enhanced state machine parameters
            attack_rate = 0.8 / (1 + np.exp(-3*(motion_state[i-1] - 0.4)))  # From 0.85
            decay_rate = 0.2 * (1 - 0.5*motion_state[i-1])  # From 0.15

            if acc_mag[i] > current_threshold * (1 + 0.3*current_noise):
                motion_state[i] = min(motion_state[i-1] + attack_rate, 1.0)
            else:
                motion_state[i] = max(motion_state[i-1] - decay_rate, 0.0)
        
        # Final quantization threshold
        motion_state = np.where(motion_state > 0.5, 1.0, 0.0)  # From 0.52
        
        dataset['motion_burst'] = motion_state
        return dataset