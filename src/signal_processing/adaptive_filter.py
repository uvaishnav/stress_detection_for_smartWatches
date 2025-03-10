import numpy as np
import logging
from scipy import signal

logging.basicConfig(level=logging.INFO)

class AdaptiveFilter:
    def __init__(self, learning_rate: float = 0.001, filter_length: int = 30):
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

    def initialize_coefficients(self):
        """Initialize/reinitialize filter coefficients with proper length"""
        self.coefficients = np.random.uniform(-0.1, 0.1, self.filter_length)

    def _bandpass_filter(self, acc_mag: np.ndarray) -> np.ndarray:
        """Add bandpass filtering to reference signal"""
        return signal.sosfilt(self.bp_filter, acc_mag)

    def apply_adaptive_filter(self, noisy_signal: np.ndarray, 
                             reference_signal: np.ndarray,
                             motion_burst: np.ndarray) -> np.ndarray:
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
        
        # Motion-aware learning rate adjustment
        motion_factor = 1 + 2*motion_burst  # Increase learning during motion
        self.learning_rate *= motion_factor
        
        # Noise-dependent regularization
        noise_power = np.mean(reference_signal**2)
        regularized_update = self.learning_rate / (1 + noise_power)
        
        # Frequency-domain processing for better convergence
        freq_signal = np.fft.fft(noisy_signal)
        freq_reference = np.fft.fft(reference_signal)
        
        # Spectral subtraction with adaptive floor
        clean_spectrum = freq_signal - 0.7*freq_reference
        clean_spectrum *= (np.abs(clean_spectrum) > 0.2*np.max(np.abs(freq_signal)))
        
        filtered_signal = np.zeros_like(noisy_signal)
        dc_offset = np.median(noisy_signal)  # Capture DC offset before filtering
        epsilon = 1e-6
        max_order = min(self.filter_length, len(noisy_signal) // 2)  # Dynamic upper limit

        # Add numerical stability measures
        EPSILON = 1e-8
        MAX_UPDATE = 0.1
        
        # Vectorized implementation replaces the for-loop
        num_samples = len(noisy_signal)
        
        # Create sliding window view of reference signal
        shape = (num_samples - max_order + 1, max_order)
        strides = (reference_signal.strides[0], reference_signal.strides[0])
        ref_windows = np.lib.stride_tricks.as_strided(reference_signal, 
                                                     shape=shape,
                                                     strides=strides)[::-1]

        # Vectorized computation
        for i in range(len(noisy_signal)):
            if i < self.filter_length:
                continue
            
            # Extract reference window
            ref_window = reference_signal[i-self.filter_length:i]
            
            # Calculate output and error
            output = np.dot(self.coefficients, ref_window)
            error = noisy_signal[i] - output
            
            # Update coefficients with regularization
            norm = np.dot(ref_window, ref_window) + EPSILON
            step_size = self.learning_rate[i] * error / norm
            self.coefficients += step_size * ref_window
            
            filtered_signal[i] = output

        if np.any(np.isnan(filtered_signal)):
            logging.warning("Adaptive filtering produced NaNs")
            
        # Post-filter restoration
        filtered_signal = filtered_signal * (np.std(noisy_signal) + 1e-9) + signal_mean
        
        # Add post-processing variation
        filtered_signal = filtered_signal + np.random.normal(0, 0.001, len(filtered_signal))  # Add micro-noise
        
        # Motion-aware stability control
        self.learning_rate = np.clip(0.01 / (1 + 2*motion_burst), 0.001, 0.1)
        
        # Physiological constraint: limit output variation
        for i in range(self.filter_length, len(noisy_signal)):
            # ... existing filter code ...
            
            # Enforce physiological plausibility
            if i > 10:
                prev_avg = np.mean(filtered_signal[i-5:i])
                filtered_signal[i] = 0.8*filtered_signal[i] + 0.2*prev_avg
        
        # Calculate noise level from reference signal
        noise_level = np.sqrt(np.mean(reference_signal**2))  # RMS of reference signal
        
        # Stabilized learning rate using calculated noise level
        base_lr = np.clip(0.1 / (1 + 5*noise_level), 0.001, 0.1)
        self.learning_rate = base_lr * (1 + 0.5*np.mean(motion_burst))
        
        # Physiological envelope constraint
        signal_envelope = np.abs(signal.hilbert(noisy_signal))
        for i in range(len(filtered_signal)):
            if filtered_signal[i] > 1.5*signal_envelope[i]:
                filtered_signal[i] = 0.8*filtered_signal[i] + 0.2*signal_envelope[i]
        
        return filtered_signal