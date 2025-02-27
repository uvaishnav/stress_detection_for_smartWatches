# src/signal_processing/wavelet_denoiser.py

import pywt
import numpy as np

class WaveletDenoiser:
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        self.wavelet = wavelet
        self.level = level

    def apply_wavelet_denoising(self, signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return np.zeros_like(signal)
        
        # Ensure input has no NaNs
        signal = np.nan_to_num(signal, nan=0.0)

        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, mode='periodization')
        except ValueError:
            return np.zeros_like(signal)

        # Handle empty coefficients or zero sigma
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        if np.isnan(sigma) or sigma == 0:
            sigma = 1e-6

        base_threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        adjusted_thresholds = []
        for i in range(len(coeffs)):
            # Ensure valid motion burst window
            start_idx = max(0, i - self.level)
            end_idx = i
            burst_window = motion_burst[start_idx:end_idx]
            
            if len(burst_window) == 0:
                motion_factor = 0.0  # Default to no motion boost
            else:
                motion_factor = np.mean(burst_window)
                if np.isnan(motion_factor):
                    motion_factor = 0.0
            
            adjusted_thresholds.append(base_threshold * (1 + 2 * motion_factor))

        # Apply soft thresholding
        coeffs_denoised = [
            pywt.threshold(c, t, mode='soft') if np.any(c) else c 
            for c, t in zip(coeffs, adjusted_thresholds)
        ]

        try:
            denoised_signal = pywt.waverec(coeffs_denoised, self.wavelet, mode='periodization')
        except ValueError:
            denoised_signal = np.zeros_like(signal)

        # Align lengths
        denoised_signal = denoised_signal[:len(signal)] if len(denoised_signal) > len(signal) else np.pad(
            denoised_signal, (0, max(0, len(signal) - len(denoised_signal))), mode='constant'
        )

        return np.nan_to_num(denoised_signal, nan=0.0)
