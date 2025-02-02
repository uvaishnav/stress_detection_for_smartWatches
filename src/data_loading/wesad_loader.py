import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .base_loader import BaseDataLoader
import os
import pickle
import gzip
from typing import Dict, Any

class WESADLoader(BaseDataLoader):
    """Loader for WESAD dataset (Wrist-worn sensor subset)
    
    Features:
    - Handles compressed (.pkl.gz) and uncompressed pickle files
    - Validates sensor data ranges and temporal consistency
    - Maintains original label semantics while resampling
    """
    
    # Dataset constants
    WRIST_DEVICE = 'wrist'
    LABEL_MAP = {
        0: 'baseline',
        1: 'stress',
        2: 'amusement',
        3: 'meditation'
    }
    SENSOR_RATES = {
        'bvp': 64,
        'acc': 32,
        'label': 700
    }


    def __init__(self, data_path: str, target_rate: int = 30):
        super().__init__(data_path, target_rate, dataset_name='wesad')
        self.raw_sample_rate = self.SENSOR_RATES['bvp']
        # Override base validation thresholds
        self.VALIDATION_CONFIG = self.validation_config

    def load_subject(self, subject_id: int) -> pd.DataFrame:
        """Load and process data for a single subject
        Args:
            subject_id: Numeric subject identifier (e.g., 2 for S2)
        Returns:
            DataFrame with aligned sensor data and labels
        """
        subj_path = Path(self.data_path) / f'S{subject_id}' / f'S{subject_id}.pkl'
        
        try:
            with self._open_file(subj_path) as f:
                raw_data = pickle.load(f, encoding='latin1')
            return self._process_subject(raw_data, subject_id)
        except Exception as e:
            self.logger.error(f"Failed loading {subj_path}: {str(e)}")
            raise
            
    def _open_file(self, path: Path):
        """Handle both gzipped and regular pickle files"""
        if path.suffix == '.gz':
            return gzip.open(path, 'rb')
        return open(path, 'rb')

    def _process_subject(self, raw_data: Dict, subject_id: int) -> pd.DataFrame:
        """Process raw subject data through pipeline"""
        try:
            # Extract and validate raw signals
            wrist_data = raw_data['signal'][self.WRIST_DEVICE]
            self._validate_raw_shapes(wrist_data, subject_id)
            
            # Create individual sensor DataFrames
            bvp_df = self._create_bvp_df(wrist_data['BVP'])
            acc_df = self._create_acc_df(wrist_data['ACC'])
            label_df = self._create_label_df(raw_data['label'])
            
            # Resample and merge
            merged = self._resample_and_merge(
                bvp_df, 
                acc_df, 
                label_df, 
                subject_id
            )
            
            # Add normalization step
            merged[self.BVP_COL] = self._normalize_bvp(merged[self.BVP_COL])
            
            self._validate_subject(merged)
            return merged
        except KeyError as e:
            raise ValueError(f"Missing expected data key: {str(e)}") from e

    def _validate_raw_shapes(self, wrist_data: Dict, subject_id: int) -> None:
        """More tolerant sample validation"""
        expected = {
            'BVP': self.SENSOR_RATES['bvp'] * 60 * 60 * 2,  # 2 hours
            'ACC': self.SENSOR_RATES['acc'] * 60 * 60 * 2
        }
        
        for sensor in ['BVP', 'ACC']:
            actual = wrist_data[sensor].size
            if actual < 0.7 * expected[sensor]:
                raise ValueError(
                    f"S{subject_id} {sensor} samples ({actual}) "
                    f"< 70% of expected ({expected[sensor]})"
                )
            elif actual != expected[sensor]:
                self.logger.info(
                    f"S{subject_id} {sensor} samples ({actual}) "
                    f"deviate from expected ({expected[sensor]})"
                )

    def _create_bvp_df(self, bvp_data: np.ndarray) -> pd.DataFrame:
        """Create BVP DataFrame with proper indexing"""
        return pd.DataFrame(
            {self.BVP_COL: bvp_data.flatten()},
            index=pd.date_range(
                start=0,
                periods=len(bvp_data),
                freq=f'{1000//self.SENSOR_RATES["bvp"]}ms'
            )
        )

    def _create_acc_df(self, acc_data: np.ndarray) -> pd.DataFrame:
        """Create 3-axis ACC DataFrame"""
        if acc_data.shape[1] != 3:
            raise ValueError(f"Unexpected ACC shape {acc_data.shape}")
            
        return pd.DataFrame(
            acc_data,
            columns=self.ACC_COLS,
            index=pd.date_range(
                start=0,
                periods=len(acc_data),
                freq=f'{1000//self.SENSOR_RATES["acc"]}ms'
            )
        )

    def _create_label_df(self, label_data: np.ndarray) -> pd.DataFrame:
        """Create label DataFrame with unknown values mapped to baseline"""
        # Clean labels before DataFrame creation
        valid_labels = set(self.LABEL_MAP.keys())
        cleaned_labels = np.array([
            0 if label not in valid_labels else label 
            for label in label_data.flatten()
        ])
        
        return pd.DataFrame(
            {'label': cleaned_labels},
            index=pd.date_range(
                start=0,
                periods=len(label_data),
                freq=f'{1000//self.SENSOR_RATES["label"]}ms'
            )
        )

    def _resample_and_merge(self, bvp_df: pd.DataFrame, acc_df: pd.DataFrame,
                          label_df: pd.DataFrame, subject_id: int) -> pd.DataFrame:
        """Resample and merge sensor streams"""
        # Resample with sensor-specific rates
        bvp_resampled = self.resample_data(bvp_df, self.SENSOR_RATES['bvp'])
        acc_resampled = self.resample_data(acc_df, self.SENSOR_RATES['acc'])
        label_resampled = self._resample_labels(label_df)
        
        # Merge components
        merged = bvp_resampled.join(acc_resampled, how='outer')
        merged['label'] = label_resampled['label']
        
        # Add metadata
        merged['subject_id'] = subject_id
        merged['sampling_rate'] = self.target_rate
        
        return merged.ffill().dropna()

    def _resample_labels(self, label_df: pd.DataFrame) -> pd.DataFrame:
        """Resample labels using forward fill"""
        return label_df.resample(
            f'{1000//self.target_rate}ms'
        ).ffill().dropna()

    def get_labels(self) -> Dict[str, Any]:
        """Return complete label documentation"""
        return {
            'classes': self.LABEL_MAP,
            'description': (
                "Experimental conditions: baseline, stress induction, "
                "amusement video, meditation period"
            ),
            'sensor_rates': self.SENSOR_RATES,
            'dataset_reference': "https://ieeexplore.ieee.org/document/8320818"
        }

    def _validate_subject(self, df: pd.DataFrame) -> None:
        # BVP range check
        bvp_min, bvp_max = self.VALIDATION_CONFIG['bvp_range']
        bvp_violations = ~df[self.BVP_COL].between(bvp_min, bvp_max)
        if bvp_violations.any():
            self.logger.warning(
                f"{bvp_violations.sum()} BVP values outside {bvp_min}-{bvp_max} range"
            )

        # ACC magnitude check
        acc_norm = np.linalg.norm(df[self.ACC_COLS], axis=1)
        acc_violations = acc_norm >= self.VALIDATION_CONFIG['acc_limit']
        if acc_violations.any():
            self.logger.warning(
                f"{acc_violations.sum()} ACC samples exceed {self.VALIDATION_CONFIG['acc_limit']}g"
            )

        # Temporal checks
        if not df.index.is_monotonic_increasing:
            self.logger.warning("Non-monotonic timestamps detected")
        
        if df.index.duplicated().any():
            self.logger.warning("Duplicate timestamps found")

        # Label validation
        valid_labels = set(self.LABEL_MAP.keys())
        present_labels = set(df['label'].unique())
        if invalid := present_labels - valid_labels:
            self.logger.warning(f"Invalid labels detected: {invalid}")

    def _normalize_bvp(self, bvp_series: pd.Series) -> pd.Series:
        """Normalize BVP signals to [-1, 1] range"""
        bvp_min = bvp_series.min()
        bvp_max = bvp_series.max()
        
        # Handle zero-division and preserve original if normalization fails
        try:
            normalized = 2 * ((bvp_series - bvp_min) / (bvp_max - bvp_min)) - 1
            self.logger.info(f"Normalized BVP range: {bvp_min:.2f}-{bvp_max:.2f} → [-1, 1]")
            return normalized
        except ZeroDivisionError:
            self.logger.warning("Could not normalize BVP - identical min/max values")
            return bvp_series 