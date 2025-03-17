import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal

from .base_feature_extractor import BaseFeatureExtractor

class ImageEncodingExtractor(BaseFeatureExtractor):
    """
    Extracts image-based encodings from physiological signals.
    
    This includes Gramian Angular Summation Field (GASF) and 
    Markov Transition Field (MTF) encodings.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30, image_size: int = 24):
        """
        Initialize the image encoding feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
            image_size: Size of the resulting image encoding (default: 24x24)
        """
        super().__init__(window_size, overlap)
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        
    def extract_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract image encoding features from a single window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        features = {}
        
        # Resample the window to the desired image size
        resampled = self._resample_window(window)
        
        # Generate GASF encoding
        gasf = self._generate_gasf(resampled)
        
        # Generate MTF encoding
        mtf = self._generate_mtf(resampled)
        
        # Extract statistical features from the encodings
        features.update(self._extract_encoding_features(gasf, 'gasf'))
        features.update(self._extract_encoding_features(mtf, 'mtf'))
        
        return features
    
    def _resample_window(self, window: np.ndarray) -> np.ndarray:
        """
        Resample the window to the desired image size.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Resampled window
        """
        # If window is too short, pad it
        if len(window) < self.image_size:
            return np.pad(window, (0, self.image_size - len(window)), 'constant')
        
        # Otherwise, resample it
        indices = np.linspace(0, len(window) - 1, self.image_size)
        indices = np.floor(indices).astype(int)
        return window[indices]
    
    def _generate_gasf(self, window: np.ndarray) -> np.ndarray:
        """
        Generate Gramian Angular Summation Field encoding.
        
        Args:
            window: Numpy array containing the resampled window
            
        Returns:
            GASF encoding as a 2D numpy array
        """
        # Normalize to [-1, 1]
        min_val = np.min(window)
        max_val = np.max(window)
        
        if max_val == min_val:
            # Handle constant signal
            normalized = np.zeros_like(window)
        else:
            normalized = 2 * ((window - min_val) / (max_val - min_val)) - 1
        
        # Convert to polar coordinates
        phi = np.arccos(normalized)
        
        # Calculate GASF
        gasf = np.zeros((len(window), len(window)))
        for i in range(len(window)):
            for j in range(len(window)):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        
        return gasf
    
    def _generate_mtf(self, window: np.ndarray) -> np.ndarray:
        """
        Generate Markov Transition Field encoding.
        
        Args:
            window: Numpy array containing the resampled window
            
        Returns:
            MTF encoding as a 2D numpy array
        """
        # Quantize the window into Q bins
        Q = 8  # Number of quantization bins
        
        # Normalize to [0, Q-1]
        min_val = np.min(window)
        max_val = np.max(window)
        
        if max_val == min_val:
            # Handle constant signal
            quantized = np.zeros_like(window, dtype=int)
        else:
            quantized = np.floor((Q - 1) * (window - min_val) / (max_val - min_val)).astype(int)
        
        # Ensure values are within [0, Q-1]
        quantized = np.clip(quantized, 0, Q - 1)
        
        # Calculate Markov transition matrix
        markov = np.zeros((Q, Q))
        for i in range(len(quantized) - 1):
            markov[quantized[i], quantized[i + 1]] += 1
        
        # Normalize rows to get transition probabilities
        row_sums = markov.sum(axis=1)
        markov = np.divide(markov, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
        
        # Generate MTF
        mtf = np.zeros((len(window), len(window)))
        for i in range(len(window)):
            for j in range(len(window)):
                mtf[i, j] = markov[quantized[i], quantized[j]]
        
        return mtf
    
    def _extract_encoding_features(self, encoding: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Extract statistical features from an encoding.
        
        Args:
            encoding: 2D numpy array containing the encoding
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Basic statistics
        features[f'{prefix}_mean'] = np.mean(encoding)
        features[f'{prefix}_std'] = np.std(encoding)
        features[f'{prefix}_min'] = np.min(encoding)
        features[f'{prefix}_max'] = np.max(encoding)
        
        # Flatten the encoding for percentiles
        flat = encoding.flatten()
        features[f'{prefix}_p25'] = np.percentile(flat, 25)
        features[f'{prefix}_p50'] = np.percentile(flat, 50)
        features[f'{prefix}_p75'] = np.percentile(flat, 75)
        
        # Entropy
        hist, _ = np.histogram(flat, bins=10, density=True)
        hist = hist[hist > 0]  # Avoid log(0)
        features[f'{prefix}_entropy'] = -np.sum(hist * np.log2(hist))
        
        # Energy
        features[f'{prefix}_energy'] = np.sum(encoding**2)
        
        # Diagonal features
        diag = np.diag(encoding)
        features[f'{prefix}_diag_mean'] = np.mean(diag)
        features[f'{prefix}_diag_std'] = np.std(diag)
        
        # Upper triangle vs lower triangle
        triu = np.triu(encoding, k=1)
        tril = np.tril(encoding, k=-1)
        features[f'{prefix}_triu_mean'] = np.mean(triu[triu != 0]) if np.any(triu != 0) else 0
        features[f'{prefix}_tril_mean'] = np.mean(tril[tril != 0]) if np.any(tril != 0) else 0
        features[f'{prefix}_asymmetry'] = features[f'{prefix}_triu_mean'] - features[f'{prefix}_tril_mean']
        
        return features
    
    def generate_image_encodings(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate both GASF and MTF encodings for visualization or CNN input.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Tuple of (GASF encoding, MTF encoding)
        """
        # Resample the window to the desired image size
        resampled = self._resample_window(window)
        
        # Generate encodings
        gasf = self._generate_gasf(resampled)
        mtf = self._generate_mtf(resampled)
        
        return gasf, mtf 