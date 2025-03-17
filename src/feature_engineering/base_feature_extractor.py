import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.
    
    This class defines the interface for all feature extractors and provides
    common functionality for windowing and feature extraction.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
    def create_windows(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create overlapping windows from the input data.
        
        Args:
            data: 1D numpy array containing the signal
            
        Returns:
            List of numpy arrays, each representing a window of the signal
        """
        # Calculate the number of windows
        n_samples = len(data)
        n_windows = max(1, (n_samples - self.window_size) // self.step_size + 1)
        
        # Create the windows
        windows = []
        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Ensure we don't go beyond the data length
            if end_idx <= n_samples:
                windows.append(data[start_idx:end_idx])
        
        return windows
    
    def create_windows_from_df(self, df: pd.DataFrame, column: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create overlapping windows from a DataFrame column.
        
        Args:
            df: DataFrame containing the data
            column: Name of the column to extract windows from
            
        Returns:
            Tuple containing:
                - List of numpy arrays, each representing a window of the signal
                - List of starting indices for each window
        """
        data = df[column].values
        n_samples = len(data)
        n_windows = max(1, (n_samples - self.window_size) // self.step_size + 1)
        
        windows = []
        start_indices = []
        
        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Ensure we don't go beyond the data length
            if end_idx <= n_samples:
                windows.append(data[start_idx:end_idx])
                start_indices.append(start_idx)
        
        return windows, start_indices
    
    def extract_features_from_windows(self, windows: List[np.ndarray]) -> pd.DataFrame:
        """
        Extract features from a list of windows.
        
        Args:
            windows: List of numpy arrays, each representing a window of the signal
            
        Returns:
            DataFrame containing the extracted features, with one row per window
        """
        features_list = []
        
        for window in windows:
            features = self.extract_features(window)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    @abstractmethod
    def extract_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single window.
        
        This method must be implemented by all subclasses.
        
        Args:
            window: Numpy array containing the signal window
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        pass
    
    def process_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Process a DataFrame by extracting features from specified columns.
        
        Args:
            df: DataFrame containing the data
            columns: List of column names to extract features from
            
        Returns:
            DataFrame containing the extracted features, with one row per window
        """
        all_features = []
        start_indices = None
        
        for column in columns:
            windows, indices = self.create_windows_from_df(df, column)
            
            # Store the start indices from the first column
            if start_indices is None:
                start_indices = indices
            
            # Extract features from each window
            features_df = self.extract_features_from_windows(windows)
            
            # Rename columns to include the source column name
            features_df = features_df.add_prefix(f"{column}_")
            
            all_features.append(features_df)
        
        # Combine all features
        result = pd.concat(all_features, axis=1)
        
        # Add window start indices
        result['window_start_idx'] = start_indices
        
        return result 