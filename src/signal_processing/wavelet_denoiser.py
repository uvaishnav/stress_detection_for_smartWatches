import pywt
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class WaveletDenoiser:
    # Mapping of skin tones to wavelet types and thresholding methods
    WAVELET_MAP = {
        'I-II': ('db8', 'universal'),
        'III-IV': ('sym6', 'sure'),
        'V-VI': ('coif3', 'bayes')
    }

    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize the WaveletDenoiser with a specific wavelet type and decomposition level.
        
        :param wavelet: Type of wavelet to use for denoising.
        :param level: Level of wavelet decomposition.
        """
        self.wavelet = wavelet
        self.level = level

    def _threshold(self, coeffs, method='universal'):
        """
        Apply thresholding to wavelet coefficients based on the specified method.
        
        :param coeffs: Wavelet coefficients.
        :param method: Thresholding method ('universal', 'sure', 'bayes').
        :return: Thresholded coefficients.
        """
        if method == 'universal':
            threshold = np.sqrt(2 * np.log(len(coeffs))) * np.median(np.abs(coeffs)) / 0.6745
        elif method == 'sure':
            threshold = np.sort(np.abs(coeffs))[int(len(coeffs) * 0.85)]
        elif method == 'bayes':
            threshold = np.sqrt(np.mean(coeffs ** 2))
        return pywt.threshold(coeffs, threshold, mode='soft')

    def apply_wavelet_denoising(self, signal: np.ndarray, motion_burst: np.ndarray, skin_tone: str) -> np.ndarray:
        """
        Apply wavelet denoising to the input signal considering motion bursts and skin tone.
        
        :param signal: Input signal to be denoised.
        :param motion_burst: Array indicating motion bursts.
        :param skin_tone: Skin tone category to determine wavelet and thresholding method.
        :return: Denoised signal.
        """
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return np.zeros_like(signal)
        
        # Replace NaNs in the signal with zeros
        signal = np.nan_to_num(signal, nan=0.0)

        # Get wavelet and thresholding method based on skin tone
        wavelet, method = self.WAVELET_MAP.get(skin_tone, (self.wavelet, 'universal'))
        
        # Determine the maximum level of decomposition
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        safe_level = min(self.level, max_level)
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=safe_level, mode='periodization')
        except ValueError:
            return np.zeros_like(signal)

        # Estimate noise standard deviation from the detail coefficients at the highest level
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        sigma = 1e-6 if np.isnan(sigma) or sigma == 0 else sigma
        
        # Adjust threshold based on motion bursts
        for i in range(1, len(coeffs)):
            burst_window = motion_burst[max(0, len(motion_burst) - len(coeffs[i])):]
            motion_factor = np.mean(burst_window) if len(burst_window) > 0 else 0
            base_thresh = sigma * np.sqrt(2 * np.log(len(signal)))
            coeffs[i] = self._threshold(coeffs[i], method) * (1 + motion_factor)
            
        try:
            # Reconstruct the denoised signal from the thresholded coefficients
            denoised_signal = pywt.waverec(coeffs, wavelet, mode='periodization')
        except ValueError:
            denoised_signal = np.zeros_like(signal)
        
        # Align the length of the denoised signal with the original signal
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        elif len(denoised_signal) < len(signal):
            denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
        
        return np.nan_to_num(denoised_signal, nan=0.0)

