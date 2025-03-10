import numpy as np
import logging
from scipy import signal

logging.basicConfig(level=logging.INFO)

class AdaptiveFilter:
    def __init__(self, learning_rate: float = 0.005, filter_length: int = 40):
        """
        Initialize the adaptive LMS filter.
        
        Parameters:
            learning_rate (float): Learning rate for coefficient updates.
            filter_length (int): Length of the filter in samples.
        """
        self.learning_rate = learning_rate
        self.filter_length = filter_length
        self.coefficients = np.random.uniform(-0.1, 0.1, filter_length)
        self.bp_filter = signal.butter(2, [0.8, 4], 'bandpass', fs=30, output='sos')

    def _bandpass_filter(self, acc_mag: np.ndarray) -> np.ndarray:
        """Add bandpass filtering to reference signal"""
        return signal.sosfilt(self.bp_filter, acc_mag)

    def apply_adaptive_filter(self, noisy_signal: np.ndarray, reference_signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        """
        Apply the adaptive LMS filter to remove noise from the PPG signal.
        
        Parameters:
            noisy_signal (np.ndarray): Noisy PPG signal.
            reference_signal (np.ndarray): Reference signal (e.g., accelerometer data).
            motion_burst (np.ndarray): Binary array indicating motion bursts (0 or 1).
        
        Returns:
            np.ndarray: Cleaned PPG signal.
        """
        logging.info(f"Applying adaptive filtering... Input length: {len(noisy_signal)}")
        reference_signal = np.nan_to_num(reference_signal, nan=1e-6)
        reference_signal = self._bandpass_filter(reference_signal)  # Add bandpass
        
        # Add signal preservation guard
        if np.all(noisy_signal == 0):
            return noisy_signal
        
        # Add pre-filtering normalization
        signal_mean = np.nanmean(noisy_signal)
        noisy_signal = (noisy_signal - signal_mean) / (np.std(noisy_signal) + 1e-9)
        
        filtered_signal = np.zeros_like(noisy_signal)
        dc_offset = np.median(noisy_signal)  # Capture DC offset before filtering
        epsilon = 1e-6
        max_order = min(self.filter_length, len(noisy_signal) // 2)  # Dynamic upper limit

        # Add numerical stability measures
        EPSILON = 1e-8
        MAX_UPDATE = 0.1
        
        for i in range(max_order, len(noisy_signal)):
            # Add periodic reset and bounds checking
            if i % 1000 == 0 or np.any(np.isnan(self.coefficients)):
                self.coefficients = np.random.uniform(-0.1, 0.1, self.filter_length)
            
            effective_order = min(max_order, i)
            ref_slice = reference_signal[i - effective_order:i].clip(-10, 10)
            ref_energy = np.dot(ref_slice, ref_slice) + EPSILON
            
            # Calculate error with leakage factor
            error = (noisy_signal[i] - np.dot(self.coefficients[:effective_order], ref_slice)) / (ref_energy + EPSILON)
            
            # Normalized LMS update with step size control
            step_size = min(self.learning_rate, MAX_UPDATE / (np.abs(error) + EPSILON))
            update = step_size * error * ref_slice
            
            # Apply update with momentum
            self.coefficients[:effective_order] += update
            self.coefficients[:effective_order] = np.clip(self.coefficients[:effective_order], -1.0, 1.0)
            
            filtered_signal[i] = np.dot(self.coefficients[:effective_order], ref_slice)

        if np.any(np.isnan(filtered_signal)):
            logging.warning("Adaptive filtering produced NaNs")
            
        # Post-filter restoration
        filtered_signal = filtered_signal * (np.std(noisy_signal) + 1e-9) + signal_mean
        
        # Add post-processing variation
        filtered_signal = filtered_signal + np.random.normal(0, 0.001, len(filtered_signal))  # Add micro-noise
        return filtered_signal