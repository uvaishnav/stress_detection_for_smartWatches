import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from pathlib import Path

class BaseDataLoader(ABC):
    """Abstract base class for physiological dataset loaders
    
    Provides common functionality for:
    - Data resampling with anti-aliasing
    - Sensor data validation
    - Configuration management
    
    Subclasses must implement:
    - load_subject() for dataset-specific loading
    - get_labels() for label documentation
    """
    
    # Common configuration defaults
    TARGET_SAMPLE_RATE = 30  # Hz
    ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
    BVP_COL = 'bvp'
    VALIDATION_CONFIG = {
        'wesad': {
            'bvp_range': (-2.5, 2.5),
            'hr_range': (25, 210),
            'acc_limit': 3.5
        },
        'physionet': {
            'bvp_range': (-1.0, 2.0),
            'hr_range': (35, 200),
            'acc_limit': 3.0
        },
        'default': {
            'bvp_range': (-1.5, 1.5),
            'hr_range': (30, 190),
            'acc_limit': 2.8
        }
    }

    def __init__(self, data_path: str, target_rate: int, dataset_name: str = 'default'):
        """Initialize base loader
        Args:
            data_path: Root directory of the dataset
            target_rate: Target sampling rate in Hz (default: 30)
            dataset_name: Name of the dataset (default: 'default')
        """
        self.data_path = Path(data_path)
        self.target_rate = target_rate
        
        
        if dataset_name not in self.VALIDATION_CONFIG:
            raise ValueError(f"Invalid dataset: {dataset_name}. Options: {list(self.VALIDATION_CONFIG.keys())}")
            
        self.validation_config = self.VALIDATION_CONFIG[dataset_name]
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def load_subject(self, subject_id: int, exam_type: str) -> pd.DataFrame:
        """Load data for a single subject with exam type
        Args:
            subject_id: Unique subject identifier
            exam_type: Type of examination (dataset-specific)
        Returns:
            DataFrame with indexed timestamps and sensor columns
        """
        pass
    
    @abstractmethod
    def get_labels(self) -> Dict[str, Any]:
        """Get label mapping documentation
        Returns:
            Dictionary containing label definitions and metadata
        """
        pass
    
    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
        """Apply anti-aliasing filter for downsampling
        Args:
            data: Input signal array
            cutoff: Cutoff frequency in Hz
            fs: Original sampling rate
            order: Filter order (default: 5)
        Returns:
            Filtered signal array
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    def resample_data(self, data: pd.DataFrame, original_rate: int) -> pd.DataFrame:
        """Resample DataFrame to target rate with proper anti-aliasing
        Args:
            data: Input DataFrame with datetime index
            original_rate: Original sampling rate in Hz
        Returns:
            Resampled DataFrame at target_rate
        """
        if original_rate == self.target_rate:
            return data.copy()
        
        # Handle downsampling with anti-alias filtering
        if original_rate > self.target_rate:
            nyquist = self.target_rate / 2
            cutoff = nyquist * 0.8  # 80% of nyquist frequency
            
            data = data.apply(
                lambda x: self.butter_lowpass_filter(x, cutoff, original_rate),
                axis=0
            )
        
        resampled = data.resample(f'{1000//self.target_rate}ms').mean()
        return resampled.ffill().bfill()

    def _validate_subject(self, data: pd.DataFrame) -> None:
        """Flexible validation that skips missing columns"""
        if self.BVP_COL in data.columns:
            bvp_min, bvp_max = self.validation_config['bvp_range']
            bvp_violations = ~data[self.BVP_COL].between(bvp_min, bvp_max)
            if bvp_violations.any():
                self.logger.warning(f"{bvp_violations.sum()} BVP values outside range")

        if 'hr' in data.columns:
            hr_min, hr_max = self.validation_config['hr_range']
            hr_violations = ~data['hr'].between(hr_min, hr_max)
            if hr_violations.any():
                self.logger.warning(f"{hr_violations.sum()} HR values outside range")

        if all(col in data.columns for col in self.ACC_COLS):
            acc_norm = np.linalg.norm(data[self.ACC_COLS], axis=1)
            acc_violations = acc_norm >= self.validation_config['acc_limit']
            if acc_violations.any():
                self.logger.warning(f"{acc_violations.sum()} ACC samples exceed limit")

        if not data.index.is_monotonic_increasing:
            self.logger.warning("Non-monotonic timestamps")
        
        if data.index.duplicated().any():
            self.logger.warning("Duplicate timestamps")

    def get_loader_config(self) -> Dict[str, Any]:
        """Get current loader configuration
        Returns:
            Dictionary containing:
            - target_rate: Current sampling rate
            - validation_rules: Active validation thresholds
        """
        return {
            'target_rate': self.target_rate,
            'validation_rules': self.validation_config.copy()
        } 