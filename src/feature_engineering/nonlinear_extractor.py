import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks

from .base_feature_extractor import BaseFeatureExtractor

class NonlinearExtractor(BaseFeatureExtractor):
    """
    Extracts non-linear features from physiological signals.
    
    This includes entropy measures, Poincaré plot features, and other
    complexity metrics.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30):
        """
        Initialize the non-linear feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
        """
        super().__init__(window_size, overlap)
        self.sampling_rate = sampling_rate
        
    def extract_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract non-linear features from a single window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        features = {}
        
        # Extract entropy features
        features.update(self._extract_entropy_features(window))
        
        # Extract Poincaré plot features
        features.update(self._extract_poincare_features(window))
        
        # Extract DFA features
        features.update(self._extract_dfa_features(window))
        
        # Extract RQA features
        features.update(self._extract_rqa_features(window))
        
        return features
    
    def _extract_entropy_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract entropy-based features from the window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of entropy features
        """
        features = {}
        
        # Sample Entropy
        features['sample_entropy'] = self._calculate_sample_entropy(window, m=2, r=0.2*np.std(window))
        
        # Approximate Entropy
        features['approximate_entropy'] = self._calculate_approximate_entropy(window, m=2, r=0.2*np.std(window))
        
        # Permutation Entropy
        features['permutation_entropy'] = self._calculate_permutation_entropy(window, order=3, delay=1)
        
        return features
    
    def _extract_poincare_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract Poincaré plot features from the window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of Poincaré plot features
        """
        features = {}
        
        # Find peaks to get RR intervals
        peaks, _ = find_peaks(window, distance=self.sampling_rate//4)
        
        if len(peaks) > 2:
            # Calculate RR intervals
            rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # in ms
            
            # Calculate Poincaré plot features
            sd1, sd2 = self._calculate_poincare_sd(rr_intervals)
            
            features['sd1'] = sd1
            features['sd2'] = sd2
            features['sd1_sd2_ratio'] = sd1 / sd2 if sd2 > 0 else 0
            features['poincare_area'] = np.pi * sd1 * sd2
            
            # Cardiac Sympathetic Index (CSI) and Cardiac Vagal Index (CVI)
            features['csi'] = sd2**2 / sd1**2 if sd1 > 0 else 0
            features['cvi'] = np.log10(sd1 * sd2) if sd1 > 0 and sd2 > 0 else 0
        else:
            # Default values if not enough peaks
            features['sd1'] = 0
            features['sd2'] = 0
            features['sd1_sd2_ratio'] = 0
            features['poincare_area'] = 0
            features['csi'] = 0
            features['cvi'] = 0
        
        return features
    
    def _extract_dfa_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract Detrended Fluctuation Analysis features from the window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of DFA features
        """
        features = {}
        
        # Calculate DFA alpha1 (short-term correlations)
        alpha1 = self._calculate_dfa(window, scales=np.arange(4, 17))
        
        # Calculate DFA alpha2 (long-term correlations)
        # Only if window is long enough
        if len(window) >= 64:
            alpha2 = self._calculate_dfa(window, scales=np.arange(16, min(65, len(window)//4)))
        else:
            alpha2 = 0
        
        features['dfa_alpha1'] = alpha1
        features['dfa_alpha2'] = alpha2
        
        return features
    
    def _extract_rqa_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract Recurrence Quantification Analysis features from the window.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary of RQA features
        """
        features = {}
        
        # Simplified RQA implementation
        # In a real implementation, you would use a dedicated RQA library
        
        # Calculate recurrence rate
        features['recurrence_rate'] = self._calculate_recurrence_rate(window)
        
        # Calculate determinism
        features['determinism'] = self._calculate_determinism(window)
        
        return features
    
    def _calculate_sample_entropy(self, window: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Sample Entropy of the signal.
        
        Args:
            window: Numpy array containing the signal window
            m: Embedding dimension
            r: Tolerance (typically 0.2 * std)
            
        Returns:
            Sample Entropy value
        """
        # Simplified implementation of Sample Entropy
        # In a real implementation, you would use a dedicated library
        
        # Ensure window has enough points
        if len(window) < 2 * m + 1:
            return 0
        
        # Create embedded vectors
        def _create_vectors(data, m):
            vectors = []
            for i in range(len(data) - m + 1):
                vectors.append(data[i:i+m])
            return np.array(vectors)
        
        # Count similar patterns
        def _count_matches(vectors, r):
            N = len(vectors)
            B = 0.0
            for i in range(N-1):
                for j in range(i+1, N):
                    if np.max(np.abs(vectors[i] - vectors[j])) < r:
                        B += 2.0  # Count each match twice (i,j) and (j,i)
            return B / (N * (N-1))
        
        # Calculate for m and m+1
        vectors_m = _create_vectors(window, m)
        vectors_m1 = _create_vectors(window, m+1)
        
        count_m = _count_matches(vectors_m, r)
        count_m1 = _count_matches(vectors_m1, r)
        
        # Calculate sample entropy
        if count_m > 0 and count_m1 > 0:
            return -np.log(count_m1 / count_m)
        else:
            return 0
    
    def _calculate_approximate_entropy(self, window: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Approximate Entropy of the signal.
        
        Args:
            window: Numpy array containing the signal window
            m: Embedding dimension
            r: Tolerance (typically 0.2 * std)
            
        Returns:
            Approximate Entropy value
        """
        # Simplified implementation of Approximate Entropy
        # Very similar to Sample Entropy but with self-matches included
        
        # Ensure window has enough points
        if len(window) < 2 * m + 1:
            return 0
        
        # Create embedded vectors
        def _create_vectors(data, m):
            vectors = []
            for i in range(len(data) - m + 1):
                vectors.append(data[i:i+m])
            return np.array(vectors)
        
        # Count similar patterns (including self-matches)
        def _count_matches(vectors, r):
            N = len(vectors)
            B = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    if np.max(np.abs(vectors[i] - vectors[j])) < r:
                        B[i] += 1
            return np.sum(np.log(B / N)) / N
        
        # Calculate for m and m+1
        vectors_m = _create_vectors(window, m)
        vectors_m1 = _create_vectors(window, m+1)
        
        phi_m = _count_matches(vectors_m, r)
        phi_m1 = _count_matches(vectors_m1, r)
        
        # Calculate approximate entropy
        return phi_m - phi_m1
    
    def _calculate_permutation_entropy(self, window: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """
        Calculate Permutation Entropy of the signal.
        
        Args:
            window: Numpy array containing the signal window
            order: Order of permutation entropy
            delay: Time delay
            
        Returns:
            Permutation Entropy value
        """
        # Ensure window has enough points
        if len(window) < order * delay:
            return 0
        
        # Create embedded vectors
        N = len(window) - (order - 1) * delay
        patterns = np.zeros(N, dtype=int)
        
        for i in range(N):
            # Extract pattern
            pattern = window[i:i + order * delay:delay]
            
            # Get the permutation order
            sorted_idx = np.argsort(pattern)
            
            # Convert to factorial number system
            patterns[i] = np.sum([sorted_idx[j] * np.math.factorial(order - j - 1) for j in range(order)])
        
        # Count occurrences of each pattern
        _, counts = np.unique(patterns, return_counts=True)
        
        # Calculate probabilities
        probs = counts / N
        
        # Calculate entropy
        return -np.sum(probs * np.log2(probs))
    
    def _calculate_poincare_sd(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Poincaré plot descriptors SD1 and SD2.
        
        Args:
            rr_intervals: Array of RR intervals
            
        Returns:
            Tuple of (SD1, SD2)
        """
        if len(rr_intervals) < 2:
            return 0, 0
        
        # Create Poincaré plot data
        x = rr_intervals[:-1]
        y = rr_intervals[1:]
        
        # Calculate SD1 and SD2
        sd1 = np.std(np.subtract(y, x)) / np.sqrt(2)
        sd2 = np.std(np.add(y, x)) / np.sqrt(2)
        
        return sd1, sd2
    
    def _calculate_dfa(self, window: np.ndarray, scales: np.ndarray) -> float:
        """
        Calculate Detrended Fluctuation Analysis scaling exponent.
        
        Args:
            window: Numpy array containing the signal window
            scales: Array of scales to use for DFA
            
        Returns:
            DFA scaling exponent
        """
        # Simplified implementation of DFA
        # In a real implementation, you would use a dedicated library
        
        # Ensure window has enough points
        if len(window) < np.max(scales) * 4:
            return 0
        
        # Integrate the signal
        y = np.cumsum(window - np.mean(window))
        
        # Calculate fluctuation for each scale
        fluctuations = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Number of segments
            n_segments = len(y) // scale
            
            if n_segments < 1:
                continue
            
            # Reshape data into segments
            segments = np.reshape(y[:n_segments * scale], (n_segments, scale))
            
            # Create time array for fitting
            time = np.arange(scale)
            
            # Calculate local trend and fluctuation
            local_fluctuations = np.zeros(n_segments)
            
            for j in range(n_segments):
                # Fit polynomial (linear detrending)
                p = np.polyfit(time, segments[j], 1)
                trend = np.polyval(p, time)
                
                # Calculate fluctuation
                local_fluctuations[j] = np.sqrt(np.mean((segments[j] - trend) ** 2))
            
            # Calculate mean fluctuation for this scale
            fluctuations[i] = np.mean(local_fluctuations)
        
        # Filter out zeros
        valid_scales = scales[fluctuations > 0]
        valid_fluctuations = fluctuations[fluctuations > 0]
        
        if len(valid_scales) < 2:
            return 0
        
        # Fit power law to calculate alpha
        log_scales = np.log10(valid_scales)
        log_fluctuations = np.log10(valid_fluctuations)
        
        # Linear fit
        p = np.polyfit(log_scales, log_fluctuations, 1)
        
        # Alpha is the slope
        alpha = p[0]
        
        return alpha
    
    def _calculate_recurrence_rate(self, window: np.ndarray) -> float:
        """
        Calculate Recurrence Rate from Recurrence Quantification Analysis.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Recurrence Rate value
        """
        # Simplified implementation of Recurrence Rate
        # In a real implementation, you would use a dedicated RQA library
        
        # Normalize the window
        normalized = (window - np.mean(window)) / np.std(window) if np.std(window) > 0 else window
        
        # Create distance matrix
        N = len(normalized)
        distances = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.abs(normalized[i] - normalized[j])
        
        # Create recurrence matrix with threshold
        threshold = 0.2 * np.std(normalized)
        recurrence = distances < threshold
        
        # Calculate recurrence rate
        return np.sum(recurrence) / (N * N)
    
    def _calculate_determinism(self, window: np.ndarray) -> float:
        """
        Calculate Determinism from Recurrence Quantification Analysis.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Determinism value
        """
        # Simplified implementation of Determinism
        # In a real implementation, you would use a dedicated RQA library
        
        # Normalize the window
        normalized = (window - np.mean(window)) / np.std(window) if np.std(window) > 0 else window
        
        # Create distance matrix
        N = len(normalized)
        distances = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.abs(normalized[i] - normalized[j])
        
        # Create recurrence matrix with threshold
        threshold = 0.2 * np.std(normalized)
        recurrence = distances < threshold
        
        # Find diagonal lines (simplified)
        min_line_length = 2
        diagonal_lengths = []
        
        for i in range(-(N-min_line_length), N-min_line_length+1):
            diagonal = np.diag(recurrence, i)
            
            # Find consecutive ones
            if len(diagonal) >= min_line_length:
                # Convert to string for easier pattern matching
                diagonal_str = ''.join(['1' if x else '0' for x in diagonal])
                
                # Find all sequences of 1s
                import re
                matches = re.finditer(r'1+', diagonal_str)
                
                for match in matches:
                    length = match.end() - match.start()
                    if length >= min_line_length:
                        diagonal_lengths.append(length)
        
        # Calculate determinism
        if np.sum(recurrence) > 0 and len(diagonal_lengths) > 0:
            return np.sum(diagonal_lengths) / np.sum(recurrence)
        else:
            return 0 