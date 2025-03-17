import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks

from .base_feature_extractor import BaseFeatureExtractor

class FrequencyDomainExtractor(BaseFeatureExtractor):
    """
    Extracts frequency-domain features from physiological signals.
    
    This includes spectral power in different frequency bands, spectral ratios,
    and other spectral characteristics.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30):
        """
        Initialize the frequency domain feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
        """
        super().__init__(window_size, overlap)
        self.sampling_rate = sampling_rate
        
        # Define frequency bands (in Hz)
        self.vlf_band = (0.0033, 0.04)  # Very Low Frequency
        self.lf_band = (0.04, 0.15)     # Low Frequency
        self.hf_band = (0.15, 0.4)      # High Frequency
        self.cardiac_band = (0.8, 4.0)  # Cardiac band (for PPG)
        
    def extract_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features from a single window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        features = {}
        
        # Detrend the signal to remove low-frequency trends
        detrended = signal.detrend(window)
        
        # Calculate power spectral density using Welch's method
        freqs, psd = welch(detrended, fs=self.sampling_rate, nperseg=min(256, len(window)))
        
        # Extract band powers
        features.update(self._extract_band_powers(freqs, psd))
        
        # Extract spectral characteristics
        features.update(self._extract_spectral_characteristics(freqs, psd))
        
        # Extract dominant frequencies
        features.update(self._extract_dominant_frequencies(freqs, psd))
        
        # Extract spectral entropy
        features['spectral_entropy'] = self._calculate_spectral_entropy(psd)
        
        return features
    
    def _extract_band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """
        Extract power in different frequency bands.
        
        Args:
            freqs: Frequency array from Welch's method
            psd: Power spectral density array from Welch's method
            
        Returns:
            Dictionary of band power features
        """
        features = {}
        
        # Calculate absolute power in each band
        vlf_power = self._band_power(freqs, psd, self.vlf_band)
        lf_power = self._band_power(freqs, psd, self.lf_band)
        hf_power = self._band_power(freqs, psd, self.hf_band)
        cardiac_power = self._band_power(freqs, psd, self.cardiac_band)
        
        # Total power (excluding VLF for short-term recordings)
        total_power = lf_power + hf_power
        
        # Store absolute powers
        features['vlf_power'] = vlf_power
        features['lf_power'] = lf_power
        features['hf_power'] = hf_power
        features['cardiac_power'] = cardiac_power
        features['total_power'] = total_power
        
        # Calculate normalized powers
        if total_power > 0:
            features['lf_power_norm'] = (lf_power / total_power) * 100
            features['hf_power_norm'] = (hf_power / total_power) * 100
        else:
            features['lf_power_norm'] = 0
            features['hf_power_norm'] = 0
        
        # Calculate power ratios
        features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
        
        return features
    
    def _extract_spectral_characteristics(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """
        Extract general characteristics of the power spectrum.
        
        Args:
            freqs: Frequency array from Welch's method
            psd: Power spectral density array from Welch's method
            
        Returns:
            Dictionary of spectral characteristic features
        """
        features = {}
        
        # Spectral centroid (weighted average of frequencies)
        if np.sum(psd) > 0:
            features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        else:
            features['spectral_centroid'] = 0
        
        # Spectral spread (variance of the spectrum around the centroid)
        if np.sum(psd) > 0 and features['spectral_centroid'] > 0:
            features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
        else:
            features['spectral_spread'] = 0
        
        # Spectral skewness (asymmetry of the spectrum)
        if np.sum(psd) > 0 and features['spectral_spread'] > 0:
            features['spectral_skewness'] = np.sum(((freqs - features['spectral_centroid']) ** 3) * psd) / (np.sum(psd) * (features['spectral_spread'] ** 3))
        else:
            features['spectral_skewness'] = 0
        
        # Spectral kurtosis (peakedness of the spectrum)
        if np.sum(psd) > 0 and features['spectral_spread'] > 0:
            features['spectral_kurtosis'] = np.sum(((freqs - features['spectral_centroid']) ** 4) * psd) / (np.sum(psd) * (features['spectral_spread'] ** 4)) - 3
        else:
            features['spectral_kurtosis'] = 0
        
        # Spectral edge frequency (frequency below which X% of the power resides)
        total_power = np.sum(psd)
        if total_power > 0:
            cumulative_power = np.cumsum(psd) / total_power
            features['spectral_edge_90'] = freqs[np.where(cumulative_power >= 0.9)[0][0]] if np.any(cumulative_power >= 0.9) else freqs[-1]
            features['spectral_edge_95'] = freqs[np.where(cumulative_power >= 0.95)[0][0]] if np.any(cumulative_power >= 0.95) else freqs[-1]
        else:
            features['spectral_edge_90'] = 0
            features['spectral_edge_95'] = 0
        
        return features
    
    def _extract_dominant_frequencies(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """
        Extract dominant frequencies from the power spectrum.
        
        Args:
            freqs: Frequency array from Welch's method
            psd: Power spectral density array from Welch's method
            
        Returns:
            Dictionary of dominant frequency features
        """
        features = {}
        
        # Find peaks in the PSD
        peaks, _ = find_peaks(psd, height=np.max(psd) * 0.1)
        
        if len(peaks) > 0:
            # Sort peaks by power
            sorted_peaks = peaks[np.argsort(-psd[peaks])]
            
            # Dominant frequency (frequency with highest power)
            features['dominant_freq'] = freqs[sorted_peaks[0]]
            features['dominant_freq_power'] = psd[sorted_peaks[0]]
            
            # Second dominant frequency (if available)
            if len(sorted_peaks) > 1:
                features['second_dominant_freq'] = freqs[sorted_peaks[1]]
                features['second_dominant_freq_power'] = psd[sorted_peaks[1]]
            else:
                features['second_dominant_freq'] = 0
                features['second_dominant_freq_power'] = 0
            
            # Dominant frequency in cardiac band
            cardiac_mask = (freqs >= self.cardiac_band[0]) & (freqs <= self.cardiac_band[1])
            if np.any(cardiac_mask):
                cardiac_psd = psd.copy()
                cardiac_psd[~cardiac_mask] = 0
                cardiac_peaks, _ = find_peaks(cardiac_psd)
                
                if len(cardiac_peaks) > 0:
                    max_cardiac_peak = cardiac_peaks[np.argmax(cardiac_psd[cardiac_peaks])]
                    features['cardiac_dominant_freq'] = freqs[max_cardiac_peak]
                    features['cardiac_dominant_freq_power'] = cardiac_psd[max_cardiac_peak]
                else:
                    features['cardiac_dominant_freq'] = 0
                    features['cardiac_dominant_freq_power'] = 0
            else:
                features['cardiac_dominant_freq'] = 0
                features['cardiac_dominant_freq_power'] = 0
        else:
            # Default values if no peaks found
            features['dominant_freq'] = 0
            features['dominant_freq_power'] = 0
            features['second_dominant_freq'] = 0
            features['second_dominant_freq_power'] = 0
            features['cardiac_dominant_freq'] = 0
            features['cardiac_dominant_freq_power'] = 0
        
        return features
    
    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, freq_band: tuple) -> float:
        """
        Calculate power in a specific frequency band.
        
        Args:
            freqs: Frequency array from Welch's method
            psd: Power spectral density array from Welch's method
            freq_band: Tuple of (low_freq, high_freq) defining the band
            
        Returns:
            Power in the specified frequency band
        """
        low, high = freq_band
        mask = (freqs >= low) & (freqs <= high)
        
        # If no frequencies in the band, return 0
        if not np.any(mask):
            return 0
        
        # Calculate power in the band (area under the PSD curve)
        return np.trapz(psd[mask], freqs[mask])
    
    def _calculate_spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Calculate the spectral entropy of the signal.
        
        Args:
            psd: Power spectral density array
            
        Returns:
            Spectral entropy value
        """
        # Normalize PSD to get probability distribution
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        
        # Calculate entropy (avoid log(0))
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Normalize by maximum entropy
        max_entropy = np.log2(len(psd_norm))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy 