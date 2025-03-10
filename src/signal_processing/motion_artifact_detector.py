# src/signal_processing/motion_artifact_detector.py

import numpy as np
import pandas as pd

class MotionArtifactDetector:
    def __init__(self, acc_threshold_factor: float = 1.3,  # Reduced from 1.5
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

    def detect_motion_bursts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Detect motion bursts in the dataset based on accelerometer data.
        """
        # 1. Remove redundant device normalization
        # Keep only essential operations:
        dataset['is_clean'] = dataset['device'].str.lower() == 'clean'
        
        # 2. Optimized vectorized scaling
        scale_map = {
            'apple_watch': 2048,
            'galaxy_watch': 1024
        }
        scales = dataset['device'].map(scale_map).fillna(512).values
        acc = dataset[['acc_x', 'acc_y', 'acc_z']].values / scales[:, None]
        
        # 3. Replace robust_normalize with faster magnitude calculation
        acc_mag = np.linalg.norm(acc, axis=1)  # Faster than manual sqrt(sum)
        dataset['acc_mag'] = (acc_mag - np.median(acc_mag)) / (np.percentile(acc_mag, 75) - np.percentile(acc_mag, 25) + 1e-9)
        
        # 4. Optimized rolling quantile calculation
        dataset['motion_burst'] = 0.0
        if not dataset['is_clean'].all():
            non_clean_mask = ~dataset['is_clean']
            acc_mag_vals = dataset.loc[non_clean_mask, 'acc_mag'].values
            window_size = int(self.sampling_rate * 1.5)  # Reduced from 2 seconds
            
            # Use stride_tricks for faster rolling windows
            shape = acc_mag_vals.shape[0] - window_size + 1, window_size
            strides = acc_mag_vals.strides * 2
            rolling_windows = np.lib.stride_tricks.as_strided(acc_mag_vals, shape=shape, strides=strides)
            
            q90 = np.quantile(rolling_windows, 0.85, axis=1)  # Lower quantile
            motion_mask = np.concatenate([
                np.zeros(window_size-1),
                acc_mag_vals[window_size-1:] > q90 * 1.15  # Increased multiplier
            ])
            
            dataset.loc[non_clean_mask, 'motion_burst'] = motion_mask[:len(non_clean_mask)]
        
        return dataset