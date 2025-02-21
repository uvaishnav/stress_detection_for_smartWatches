import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from data_loading.base_loader import BaseDataLoader
from .sensor_alignment import SensorAligner
from .noise_simulator import NoiseSimulator
from abc import ABC

class BasePreparer(BaseDataLoader):
    """Handles core data preparation logic across datasets"""
    
    # Common configuration
    TARGET_LABELS = {
        0: 'baseline',
        1: 'stress',
        2: 'amusement',
        3: 'meditation'
    }
    
    def __init__(self, data_path: str, output_dir: str = "data/processed"):
        """
        Args:
            data_path: Path to raw data
            output_dir: Where to save processed files
        """
        super().__init__(data_path, target_rate=30)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.aligner = SensorAligner()
        self.current_subject = None
        self.ACC_COLS = ['acc_x', 'acc_y', 'acc_z']

    def map_labels(self, raw_labels: pd.Series, dataset: str) -> pd.Series:
        """Convert dataset-specific labels to unified scheme"""
        label_map = {
            'wesad': {
                0: 0,  # Baseline
                1: 1,  # Stress
                2: 2,  # Amusement
                3: 3   # Meditation
            },
            'physionet': {
                0: 0,  # No event → Baseline
                1: 1   # Event → Stress
            }
        }
        
        if dataset not in label_map:
            raise ValueError(f"Invalid dataset: {dataset}. Choose: {list(label_map.keys())}")
            
        return raw_labels.map(label_map[dataset]).astype('int8')

    def remap_sensors(self, data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Align sensor axes across datasets"""
        # This needs dataset-specific implementation
        raise NotImplementedError("Implement in subclass")

    def add_basic_noise(self, data: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        """Case-insensitive column access"""
        noisy_data = data.copy()
        for col in ['bvp', 'acc_x', 'acc_y', 'acc_z']:
            col_exists = [c for c in data.columns if c.lower() == col.lower()]
            if col_exists:
                actual_col = col_exists[0]
                noise = np.random.normal(0, noise_level, size=len(noisy_data))
                noisy_data[actual_col] += noise
        return noisy_data

    def save_processed(self, df: pd.DataFrame, filename: str):
        # Force skin_tone column existence
        if 'skin_tone' not in df.columns:
            df['skin_tone'] = 'none'  # Default for clean
        
        # Add skin_tone to filename template
        filename = f"{df['device'].iloc[0]}_{df['skin_tone'].iloc[0]}_{filename}"
        
        # Enforce unified schema
        required_cols = [
            'bvp', 'acc_x', 'acc_y', 'acc_z', 
            'label', 'noise_level', 'dataset',
            'subject_id', 'device', 'skin_tone'
        ]
        
        # Add missing columns with defaults
        df['dataset'] = df.get('dataset', 'unknown')
        df['device'] = df.get('device', 'clean')
        df['noise_level'] = df.get('noise_level', 0.0)
        df['skin_tone'] = df.get('skin_tone', 'clean')
        
        # Remove non-standard columns
        extra_cols = set(df.columns) - set(required_cols)
        df = df.drop(columns=list(extra_cols))
        
        # Type enforcement
        df = df.astype({
            'bvp': 'float32',
            'acc_x': 'float32',
            'acc_y': 'float32',
            'acc_z': 'float32',
            'label': 'int8',
            'noise_level': 'float32',
            'subject_id': 'uint16',
            'dataset': 'category',
            'device': 'category',
            'skin_tone': 'category'
        })
        
        # Null check
        if df.isnull().sum().sum() > 0:
            raise ValueError("Null values detected in final output")
        
        df.to_parquet(f"{self.output_dir}/{filename}.parquet")

    def validate_output(self, data: pd.DataFrame) -> bool:
        """Basic validation checks"""
        checks = [
            ('labels', lambda: data['label'].isin(self.TARGET_LABELS.keys()).all()),
            ('timestamps', lambda: data.index.is_monotonic_increasing),
        ]
        
        for name, check in checks:
            if not check():
                self.logger.error(f"Validation failed: {name}")
                return False
        return True

    def process_subject(self, subject_id: int) -> Dict[str, Any]:
        try:
            # 1. Load raw data
            raw_df = self.load_subject(subject_id)
            
            # Clip ACC values before any processing
            if {'acc_x', 'acc_y', 'acc_z'}.issubset(raw_df.columns):
                raw_df[['acc_x', 'acc_y', 'acc_z']] = raw_df[['acc_x', 'acc_y', 'acc_z']].clip(-3.5, 3.5)
            
            # 2. Sensor alignment
            aligned_ppg = self.aligner.align_ppg(raw_df, self.dataset_name)
            aligned_acc = self.aligner.align_acc(aligned_ppg, self.dataset_name)
            
            # 3. Label mapping
            aligned_acc['label'] = self.map_labels(aligned_acc['label'], self.dataset_name)
            
            # 4. Add dataset identifier
            aligned_acc['dataset'] = self.dataset_name
            
            # 5. Validate
            if not self.validate_output(aligned_acc):
                raise ValueError("Post-alignment validation failed")
            
            # 6. Generate filename
            filename = f"{self.dataset_name}_subject_{subject_id}"
            
            # 7. Save clean version
            self.save_processed(aligned_acc, f"clean_{filename}")
            
            # 8. Add noise variants
            noisy_sim = NoiseSimulator()
            
            # Add demographic-aware noise
            demographics = self.load_demographics(subject_id)  # To implement
            skin_tones = ['I-II', 'III-IV', 'V-VI']  # Fitzpatrick scale
            
            for device in ['apple_watch', 'galaxy_watch']:
                for skin in skin_tones:
                    if device == 'clean':
                        # Save original without noise
                        df = aligned_acc.copy()
                        df['device'] = 'clean'
                        df['skin_tone'] = 'none'
                        self.save_processed(df, f"clean_{filename}")
                    else:
                        # Add device+skin noise
                        noisy_df = noisy_sim.add_device_noise(
                            aligned_acc, 
                            device=device,
                            skin_tone=skin  # Synthetic skin tone
                        )
                        noisy_df['skin_tone'] = skin
                        self.save_processed(noisy_df, f"{device}_{skin}_{filename}")
            
            return {'status': 'success', 'subject': subject_id}
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {'status': 'error', 'subject': subject_id, 'error': str(e)}

    def add_noise_level(self, df: pd.DataFrame, device: str) -> pd.DataFrame:
        # Initialize base noise level (device-specific)
        base_noise = {
            'clean': 0.0,
            'apple_watch': 0.35,  # Base device noise
            'galaxy_watch': 0.25   # Base device noise
        }
        df['noise_level'] = base_noise[device]
        return df

    def _validate_output_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dataset_name == 'physionet':
            required_columns = ['hr', 'eda', 'temp', 'bvp', 'label']  # Lowercase
        else:
            required_columns = ['bvp', 'acc_x', 'acc_y', 'acc_z', 'label']
        
        # Case-insensitive check
        missing = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df

    def load_subject(self, subject_id: int, *args) -> Any:
        data = self._load_raw_data(subject_id, *args)
        if not self._validate_raw_data(data):
            raise ValueError("Invalid raw data structure")
        return data

    def _validate_raw_data(self, data: Any) -> bool:
        """Case-insensitive column check"""
        if self.dataset_name == 'wesad':
            required = {'bvp', 'acc', 'label'}
            return isinstance(data, pd.DataFrame) and required.issubset(data.columns.str.lower())
        elif self.dataset_name == 'physionet':
            required = {'hr', 'eda', 'temp', 'event', 'acc_x', 'acc_y', 'acc_z'}
            return isinstance(data, pd.DataFrame) and required.issubset(data.columns.str.lower())
        return False

    def _validate_sample_rate(self, df: pd.DataFrame) -> bool:
        """Validate sampling rate through index intervals"""
        if len(df) < 2:
            return True # Not enough samples to check
        
        # Calculate time differences in milliseconds
        diffs = df.index.to_series().diff().dt.total_seconds() * 1000
        expected_interval = 1000 / 30  # 33.333ms for 30Hz
        
        # Allow 1% tolerance
        return diffs.dropna().between(
            expected_interval * 0.99,
            expected_interval * 1.01
        ).all()
