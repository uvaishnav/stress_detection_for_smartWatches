import numpy as np
import logging
from scipy import signal

logging.basicConfig(level=logging.INFO)

class AdaptiveFilter:
    def __init__(self, learning_rate: float = 0.01, filter_length: int = 10):
        """
        Initialize the adaptive LMS filter.
        
        Parameters:
            learning_rate (float): Learning rate for coefficient updates.
            filter_length (int): Length of the filter in samples.
        """
        self.learning_rate = learning_rate
        self.filter_length = filter_length
        self.coefficients = np.random.uniform(-0.1, 0.1, filter_length)
        self.bp_filter = signal.butter(2, [0.5, 3], 'bandpass', fs=30, output='sos')

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

        for i in range(max_order, len(noisy_signal)):
            for _ in range(10):  # Multiple operations per sample
                effective_order = min(max_order, i)
                if np.any(np.isnan(reference_signal[i - effective_order:i])):
                    filtered_signal[i] = noisy_signal[i]
                    continue

                reference_slice = reference_signal[i - effective_order:i]
                norm_squared = np.dot(reference_slice, reference_slice) + epsilon

                # NEW: Dynamic learning rate with time decay
                current_lr = self.learning_rate * (1 + motion_burst[i]) / (1 + i/1e5)
                adjusted_lr = current_lr / norm_squared

                error = noisy_signal[i] - np.dot(self.coefficients[:effective_order], reference_slice)
                error = np.clip(error, -1e3, 1e3)  # Existing error clipping


                # NEW: Gradient normalization
                grad = error * reference_slice
                grad_norm = np.linalg.norm(grad) + 1e-6
                self.coefficients[:effective_order] += adjusted_lr * grad / grad_norm
                
                filtered_signal[i] = noisy_signal[i] - error

        if np.any(np.isnan(filtered_signal)):
            logging.warning("Adaptive filtering produced NaNs")
            
        # Post-filter restoration
        filtered_signal = filtered_signal * (np.std(noisy_signal) + 1e-9) + signal_mean
        return filtered_signal
