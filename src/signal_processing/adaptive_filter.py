import numpy as np
import logging
from scipy import signal
from scipy.signal import butter, sosfilt

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
        # logging.info(f"Applying adaptive filtering... Input length: {len(noisy_signal)}")
        reference_signal = np.nan_to_num(reference_signal, nan=1e-6)
        reference_signal = self._bandpass_filter(reference_signal)  # Add bandpass
        
        # Add pre-processing validation
        if np.all(noisy_signal == 0) or np.std(noisy_signal) < 1e-3:
            return noisy_signal  # Return raw signal if input is invalid
        
        # Add pre-filtering normalization
        signal_mean = np.nanmean(noisy_signal)
        noisy_signal = (noisy_signal - signal_mean) / (np.std(noisy_signal) + 1e-9)
        
        # Frequency-domain processing for better convergence
        freq_signal = np.fft.fft(noisy_signal)
        freq_reference = np.fft.fft(reference_signal)
        
        # Virtually eliminate spectral subtraction
        sub_ratio = 0.00005 + 0.0005*motion_burst  # Dramatically reduced from 0.0001 + 0.001
        
        # Motion-adaptive spectral subtraction
        clean_spectrum = freq_signal - sub_ratio*freq_reference
        
        # Add notch filter preservation
        notch_freqs = [0.8, 4]  # Preserve pulse band
        clean_spectrum = self.apply_notch_preservation(clean_spectrum, notch_freqs)
        
        filtered_signal = np.zeros_like(noisy_signal)
        dc_offset = np.median(noisy_signal)  # Capture DC offset before filtering
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

        # Motion-aware learning rate adjustment (remove array operation)
        base_learning_rate = self.learning_rate  # Store original scalar value

        # Physiological envelope constraint
        signal_envelope = np.abs(signal.hilbert(noisy_signal))
        
        # Vectorized computation using precomputed windows
        for i in range(self.filter_length, len(noisy_signal)):
            # Get precomputed reference window
            ref_window = ref_windows[i - self.filter_length]
            
            # Calculate output and error
            output = np.dot(self.coefficients, ref_window)
            error = noisy_signal[i] - output
            
            # Update coefficients with scalar operations
            norm = np.dot(ref_window, ref_window) + EPSILON
            current_lr = base_learning_rate * (1 + 2.5*motion_burst[i])  # Reduced motion sensitivity
            step_size = current_lr * error / norm
            
            # Enhanced coefficient update with gradient clipping
            update = step_size * ref_window.astype(self.coefficients.dtype)
            update = np.clip(update, -MAX_UPDATE, MAX_UPDATE)
            self.coefficients += update
            
            filtered_signal[i] = output
            
            # Enforce physiological plausibility directly in main loop
            if i > self.filter_length + 10:
                # Moving average of previous 5 samples
                prev_avg = np.mean(filtered_signal[i-5:i])
                # Weighted combination of current and historical average
                filtered_signal[i] = 0.995*filtered_signal[i] + 0.005*prev_avg  # From 0.98/0.02
                # Apply envelope constraint
                if filtered_signal[i] > 2.5*signal_envelope[i]:  # From 1.8
                    filtered_signal[i] = 0.995*filtered_signal[i] + 0.005*signal_envelope[i]  # From 0.98/0.02

            # Add periodic coefficient reset
            if i % 1000 == 0 and np.mean(np.abs(self.coefficients)) < 1e-3:
                self.initialize_coefficients()
                logging.debug("Reset adaptive filter coefficients")

        if np.any(np.isnan(filtered_signal)):
            logging.warning("Adaptive filtering produced NaNs")
            
        # Extract cardiac component for preservation
        sos = butter(3, [0.8, 4.0], btype='bandpass', fs=30, output='sos')
        cardiac_component = sosfilt(sos, noisy_signal)
        
        # Massive amplitude preservation
        filtered_signal = filtered_signal * (10.0*np.std(noisy_signal) + 1e-9) + signal_mean
        
        # Blend with original signal and enhanced cardiac component
        filtered_signal = (
            0.2 * filtered_signal + 
            0.6 * noisy_signal + 
            0.2 * (cardiac_component * 2.0)  # Boosted cardiac component
        )
        
        return filtered_signal

    def apply_notch_preservation(self, spectrum: np.ndarray, notch_freqs: list) -> np.ndarray:
        """Apply notch filter preservation to the spectrum"""
        for freq in notch_freqs:
            start_bin = int((freq-0.2) * len(spectrum)/30)
            end_bin = int((freq+0.2) * len(spectrum)/30)
            spectrum[start_bin:end_bin] *= 0.2  # Attenuate instead of zeroing
        return spectrum