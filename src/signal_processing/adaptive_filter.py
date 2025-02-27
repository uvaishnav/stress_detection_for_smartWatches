import numpy as np
import logging

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
        self.coefficients = np.zeros(filter_length)   # Small random initialization

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
        # Replace NaNs in reference_signal with small values
        reference_signal = np.nan_to_num(reference_signal, nan=1e-6)

        filtered_signal = np.zeros_like(noisy_signal)
        epsilon = 1e-6  # Small constant to avoid division by zero

        for i in range(self.filter_length, len(noisy_signal)):
            # Skip invalid indices
            if i % 100000 == 0:  # Log every 1000th iteration
                logging.info(f"Coeffs: {self.coefficients}, Error: {error}")

            if np.any(np.isnan(reference_signal[i - self.filter_length:i])):
                filtered_signal[i] = noisy_signal[i]  # Pass through raw signal
                continue

            reference_slice = reference_signal[i - self.filter_length:i]
            reference_slice = reference_slice / (np.linalg.norm(reference_slice) + 1e-6)  # Normalize input window
            norm_squared = np.dot(reference_slice, reference_slice) + epsilon
            adjusted_lr = (self.learning_rate / norm_squared) * (1 + 2 * motion_burst[i])
            error = noisy_signal[i] - np.dot(self.coefficients, reference_slice)
            error = np.clip(error, -1e3, 1e3)  # Clip large errors

            # Prevent NaNs in coefficient updates
            if not np.isnan(error):
                self.coefficients += adjusted_lr * error * reference_slice

            filtered_signal[i] = noisy_signal[i] - error

        if np.any(np.isnan(filtered_signal)):
            logging.warning("Adaptive filtering produced NaNs")

        return filtered_signal
