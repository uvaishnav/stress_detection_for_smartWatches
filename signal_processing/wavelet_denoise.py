import pywt
import numpy as np

class WaveletProcessor:
    """Skin-tone adaptive wavelet denoising"""
    
    WAVELET_MAP = {
        'I-II': ('db8', 'universal'),
        'III-IV': ('sym6', 'sure'),
        'V-VI': ('coif3', 'bayes')
    }
    
    def __init__(self, wavelet_level=5):
        self.wavelet_level = wavelet_level
        
    def _threshold(self, coeffs, method='universal'):
        """Apply thresholding based on skin tone"""
        if method == 'universal':
            threshold = np.sqrt(2 * np.log(len(coeffs))) * np.median(np.abs(coeffs)) / 0.6745
        elif method == 'sure':
            threshold = np.sort(np.abs(coeffs))[int(len(coeffs)*0.85)]
        elif method == 'bayes':
            threshold = np.sqrt(np.mean(coeffs**2))
        return pywt.threshold(coeffs, threshold, mode='soft')
    
    def denoise(self, signal: np.ndarray, skin_tone: str) -> np.ndarray:
        """Main denoising workflow with safe level calculation"""
        wavelet, method = self.WAVELET_MAP[skin_tone]
        
        # Calculate maximum safe decomposition level
        max_level = pywt.dwt_max_level(len(signal), wavelet) 
        safe_level = min(self.wavelet_level, max_level)
        
        coeffs = pywt.wavedec(signal, wavelet, level=safe_level)
        coeffs[1:] = [self._threshold(c, method) for c in coeffs[1:]]
        
        return pywt.waverec(coeffs, wavelet)[:len(signal)]
