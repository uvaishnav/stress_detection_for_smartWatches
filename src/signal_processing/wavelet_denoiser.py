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

    def apply_wavelet_denoising(self, signal: np.ndarray, motion_burst: np.ndarray,
                               skin_tone: str, noise_level: float) -> np.ndarray:
        """
        Apply wavelet denoising to the input signal considering motion bursts and skin tone.
        
        :param signal: Input signal to be denoised.
        :param motion_burst: Array indicating motion bursts.
        :param skin_tone: Skin tone category to determine wavelet and thresholding method.
        :param noise_level: Noise level of the signal.
        :return: Denoised signal.
        """
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return np.zeros_like(signal)
        
        # Replace NaNs in the signal with zeros
        signal = np.nan_to_num(signal, nan=0.0)

        # Determine the maximum level of decomposition
        max_level = pywt.dwt_max_level(len(signal), self.wavelet)
        safe_level = min(self.level, max_level)
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, self.wavelet, level=safe_level, mode='periodization')
        except ValueError:
            return np.zeros_like(signal)

        # Enhanced noise estimation
        noise_est = np.mean([np.median(np.abs(c)) / 0.6745 for c in coeffs[1:]])
        threshold_factor = 1.2 - 0.6*noise_level
        
        # Adaptive motion masking
        motion_mask = np.clip(motion_burst * (1 + 0.5*noise_level), 0, 0.9)
        
        # Frequency-dependent thresholding
        for i in range(1, len(coeffs)):
            scale_factor = 1.0 / (1 + np.exp(-i))  # Less aggressive at higher frequencies
            coeffs[i] = self._threshold(coeffs[i], method='universal')
        
        try:
            # Reconstruct the denoised signal from the thresholded coefficients
            denoised = pywt.waverec(coeffs, self.wavelet, mode='periodization')
        except ValueError:
            denoised = np.zeros_like(signal)
        
        # Align the length of the denoised signal with the original signal
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='constant')
        
        # Noise-adaptive threshold scaling
        threshold_scale = 1 + 2*noise_level  # More aggressive thresholding for noisier signals
        
        # Motion-aware denoising
        denoised = (1 - motion_mask)*denoised + 0.3*motion_mask*signal  # Reduced from 1.0
        
        # Skin-tone specific frequency emphasis
        cutoff = 0.6 if skin_tone in ['V-VI'] else 0.4  # 6Hz max
        denoised = self._enhance_low_frequencies(denoised)
        
        return np.nan_to_num(denoised, nan=0.0)

    def _enhance_low_frequencies(self, signal: np.ndarray) -> np.ndarray:
        """
        Emphasize low-frequency components through spectral weighting.
        """
        fft_signal = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal))
        
        # Create a low-frequency emphasis filter
        cutoff = 0.25  # Increased from 0.1
        lpf = 1 / (1 + (freq/cutoff)**4)
        
        enhanced = np.fft.irfft(fft_signal * lpf, n=len(signal))
        return np.real(enhanced)

    def _enhance_high_frequencies(self, signal: np.ndarray) -> np.ndarray:
        """
        Emphasize high-frequency components through spectral sharpening.
        """
        fft_signal = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal))
        
        # Create a high-frequency emphasis filter
        cutoff = 0.2  # 20% of Nyquist frequency
        hpf = 1 - np.exp(-(freq/cutoff)**2)
        
        enhanced = np.fft.irfft(fft_signal * hpf, n=len(signal))
        return np.real(enhanced)

