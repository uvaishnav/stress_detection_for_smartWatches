import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
from scipy.signal import find_peaks

from .base_feature_extractor import BaseFeatureExtractor

class TimeDomainExtractor(BaseFeatureExtractor):
    """
    Extracts time-domain features from physiological signals.
    
    This includes statistical features (mean, std, etc.) and 
    heart rate variability (HRV) features.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30):
        """
        Initialize the time domain feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
        """
        super().__init__(window_size, overlap)
        self.sampling_rate = sampling_rate
        
    def extract_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from a single window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        features = {}
        
        # Basic statistical features
        features.update(self._extract_statistical_features(window))
        
        # Peak-based features
        features.update(self._extract_peak_features(window))
        
        # HRV features (if peaks are detected)
        hrv_features = self._extract_hrv_features(window)
        if hrv_features:
            features.update(hrv_features)
        
        return features
    
    def _extract_statistical_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from the window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Central tendency
        features['mean'] = np.mean(window)
        features['median'] = np.median(window)
        
        # Dispersion
        features['std'] = np.std(window)
        features['var'] = np.var(window)
        features['range'] = np.max(window) - np.min(window)
        features['iqr'] = np.percentile(window, 75) - np.percentile(window, 25)
        features['mad'] = np.mean(np.abs(window - features['mean']))
        
        # Shape
        features['skew'] = stats.skew(window)
        features['kurtosis'] = stats.kurtosis(window)
        
        # Extrema
        features['min'] = np.min(window)
        features['max'] = np.max(window)
        features['peak_to_peak'] = features['max'] - features['min']
        
        # Percentiles
        features['p10'] = np.percentile(window, 10)
        features['p25'] = np.percentile(window, 25)
        features['p75'] = np.percentile(window, 75)
        features['p90'] = np.percentile(window, 90)
        
        # Rate of change
        if len(window) > 1:
            # First derivative
            first_derivative = np.diff(window)
            features['mean_derivative'] = np.mean(first_derivative)
            features['std_derivative'] = np.std(first_derivative)
            features['max_derivative'] = np.max(np.abs(first_derivative))
            
            # Second derivative
            if len(window) > 2:
                second_derivative = np.diff(first_derivative)
                features['mean_2nd_derivative'] = np.mean(second_derivative)
                features['std_2nd_derivative'] = np.std(second_derivative)
                features['max_2nd_derivative'] = np.max(np.abs(second_derivative))
        
        return features
    
    def _extract_peak_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract features based on signal peaks.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of peak-based features
        """
        features = {}
        
        # Find peaks
        peaks, _ = find_peaks(window, distance=self.sampling_rate//4)
        
        # Count peaks
        features['peak_count'] = len(peaks)
        
        if len(peaks) > 0:
            # Peak heights
            peak_heights = window[peaks]
            features['mean_peak_height'] = np.mean(peak_heights)
            features['std_peak_height'] = np.std(peak_heights) if len(peaks) > 1 else 0
            
            # Peak intervals
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / self.sampling_rate  # in seconds
                features['mean_peak_interval'] = np.mean(peak_intervals)
                features['std_peak_interval'] = np.std(peak_intervals)
                features['min_peak_interval'] = np.min(peak_intervals)
                features['max_peak_interval'] = np.max(peak_intervals)
                
                # Estimate heart rate
                features['est_heart_rate'] = 60 / features['mean_peak_interval']
        else:
            # Default values if no peaks found
            features['mean_peak_height'] = 0
            features['std_peak_height'] = 0
            features['mean_peak_interval'] = 0
            features['std_peak_interval'] = 0
            features['min_peak_interval'] = 0
            features['max_peak_interval'] = 0
            features['est_heart_rate'] = 0
        
        return features
    
    def _extract_hrv_features(self, window: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract heart rate variability features.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of HRV features or None if insufficient peaks
        """
        # Find peaks (potential heartbeats)
        peaks, _ = find_peaks(window, distance=self.sampling_rate//4)
        
        # Need at least 2 peaks to calculate intervals
        if len(peaks) < 2:
            return None
        
        features = {}
        
        # Calculate RR intervals (in ms)
        rr_intervals = np.diff(peaks) * (1000 / self.sampling_rate)
        
        # SDNN - Standard deviation of NN intervals
        features['sdnn'] = np.std(rr_intervals)
        
        # RMSSD - Root mean square of successive differences
        if len(rr_intervals) > 1:
            successive_diffs = np.diff(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(successive_diffs**2))
            
            # pNN50 - Percentage of successive RR intervals that differ by more than 50 ms
            nn50 = np.sum(np.abs(successive_diffs) > 50)
            features['pnn50'] = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
            
            # pNN20 - Percentage of successive RR intervals that differ by more than 20 ms
            nn20 = np.sum(np.abs(successive_diffs) > 20)
            features['pnn20'] = (nn20 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
        else:
            features['rmssd'] = 0
            features['pnn50'] = 0
            features['pnn20'] = 0
        
        # CV - Coefficient of variation
        features['hrv_cv'] = (features['sdnn'] / np.mean(rr_intervals)) * 100 if np.mean(rr_intervals) > 0 else 0
        
        return features 