import pywt
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, butter, sosfilt

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
        # Handle empty input
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return np.zeros_like(signal)
        
        # Replace NaNs in the signal with zeros
        signal = np.nan_to_num(signal, nan=0.0)
        
        # Check if input is a single sample or an array
        if len(signal.shape) == 1:
            # Process the entire array at once
            return self._process_signal_batch(signal, motion_burst, skin_tone, noise_level)
        else:
            # Process each sample individually (legacy support)
            denoised = np.zeros_like(signal)
            for i in range(len(signal)):
                denoised[i] = self._process_signal_batch(
                    signal[i:i+1], 
                    motion_burst[i:i+1], 
                    skin_tone, 
                    noise_level
                )
            return denoised

    def _process_signal_batch(self, signal: np.ndarray, motion_burst: np.ndarray,
                             skin_tone: str, noise_level: float) -> np.ndarray:
        """Process a batch of signal data with wavelet denoising"""
        # Determine the maximum level of decomposition
        max_level = pywt.dwt_max_level(len(signal), self.wavelet)
        safe_level = min(self.level, max_level)
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, self.wavelet, level=safe_level, mode='periodization')
        except ValueError:
            return np.zeros_like(signal)

        # Actual noise estimation usage
        noise_est = np.mean([np.median(np.abs(c)) / 0.6745 for c in coeffs[1:]])
        threshold_factor = 0.999 - 0.0002*noise_level  # Further increased from 0.998 - 0.0005
        
        # Handle motion_burst as array or scalar
        if isinstance(motion_burst, np.ndarray) and len(motion_burst) > 0:
            motion_value = np.mean(motion_burst)
        else:
            motion_value = motion_burst
        
        motion_mask = np.clip(motion_value * (1 + 0.5*noise_level), 0, 0.9)  # Reduced from 0.9/0.98
        threshold_scale = 1 + 0.2*noise_level  # Further reduced from 0.3

        # Extract cardiac component before thresholding
        sos = butter(4, [0.9, 3.0], btype='bandpass', fs=30, output='sos')
        cardiac = sosfilt(sos, signal)
        
        # Find peaks in cardiac component
        peaks, _ = find_peaks(cardiac, distance=15, prominence=0.1)
        
        # Create a cardiac mask for wavelet coefficients
        cardiac_mask = np.zeros_like(signal)
        if len(peaks) > 2:
            # Calculate average peak-to-peak interval
            peak_intervals = np.diff(peaks)
            avg_interval = np.mean(peak_intervals)
            
            # Create a pulse enhancement window
            window_width = int(avg_interval * 0.8)
            if window_width > 2:
                for p in peaks:
                    if p > window_width and p < len(cardiac) - window_width:
                        # Apply a Gaussian window around each peak
                        window = np.exp(-0.5 * ((np.arange(-window_width, window_width) / (window_width/2))**2))
                        cardiac_mask[p-window_width:p+window_width] = window
        
        # Apply more selective thresholding
        for i in range(1, len(coeffs)):
            # Use cardiac-aware thresholding
            if i <= 2:  # Lower frequency bands - preserve cardiac
                scale_factor = 0.5  # Less aggressive for cardiac bands
            else:
                scale_factor = 1.0 / (1 + np.exp(-(i-2)))  # More aggressive for higher bands
            
            # Use skin-tone appropriate thresholding method
            wavelet_type, thresh_method = self.WAVELET_MAP.get(skin_tone, ('db8', 'universal'))
            
            # Apply thresholding with cardiac preservation
            coeffs[i] = self._threshold(
                coeffs[i] * scale_factor,  # Apply frequency scaling
                method=thresh_method
            ) * threshold_scale  # Apply noise-adaptive scaling

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
        
        # Enhanced signal preservation with cardiac awareness
        # Use cardiac_mask to preserve cardiac regions
        cardiac_preservation = np.ones_like(signal)
        cardiac_preservation = cardiac_preservation - 0.7 * cardiac_mask  # Preserve cardiac regions
        
        denoised = (1 - 0.2*motion_mask*cardiac_preservation)*denoised + (0.7 + 0.3*noise_est)*motion_mask*cardiac_preservation*signal
        
        # Enhance cardiac component
        enhanced_cardiac = cardiac * 2.5  # Boost cardiac component
        
        # Blend with enhanced cardiac component - REDUCE AMPLIFICATION
        denoised = 0.15*denoised + 0.55*signal + 0.3*enhanced_cardiac
        
        # Dynamic cutoff application
        distance = int(30/(1 + 2*noise_level))
        peaks, _ = find_peaks(signal, distance=distance)
        pulse_rate = 60 * 30 / np.mean(np.diff(peaks)) if len(peaks)>1 else 60
        cutoff = max(0.8, pulse_rate/60 * 0.5)
        denoised = self._enhance_low_frequencies(denoised, cutoff)  # Pass dynamic cutoff
        
        # Add amplitude constraint
        if np.std(denoised) > 1.05 * np.std(signal):  # Allow only 5% increase
            denoised = (denoised - np.mean(denoised)) * (1.05 * np.std(signal) / np.std(denoised)) + np.mean(denoised)
        
        return np.nan_to_num(denoised, nan=0.0)

    def _enhance_low_frequencies(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Now uses dynamic cutoff parameter"""
        fft_signal = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal))
        lpf = 1 / (1 + (freq/cutoff)**4)  # Use passed cutoff value
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

