import pandas as pd
from pathlib import Path
from typing import Dict, Any
from data_loading.wesad_loader import WESADLoader
from .base_preparer import BasePreparer
from data_preparation.noise_simulator import NoiseSimulator
import logging
import numpy as np

class WESADPreparer(BasePreparer):
    """Handles WESAD-specific preparation tasks"""
    
    def __init__(self, data_path: str, output_dir: str = "data/processed"):
        super().__init__(data_path, output_dir)
        self.loader = WESADLoader(data_path)
        self.logger = logging.getLogger('WESADPreparer')
        self.dataset_name = 'wesad'

    def remap_sensors(self, data: pd.DataFrame) -> pd.DataFrame:
        """WESAD-specific sensor adjustments"""
        # WESAD ACC is already in ENU orientation (x=right, y=forward, z=up)
        # No remapping needed - just validate
        if not {'acc_x', 'acc_y', 'acc_z'}.issubset(data.columns):
            self.logger.error("Missing ACC axes in WESAD data")
            raise ValueError("Invalid WESAD ACC data")
            
        return data

    def process_subject(self, subject_id: int) -> Dict[str, Any]:
        """Full processing pipeline for a single subject"""
        try:
            self.current_subject = subject_id
            raw_df = self.loader.load_subject(subject_id)
            
            # Use actual columns from loaded data
            required_cols = ['bvp', 'acc_x', 'acc_y', 'acc_z', 'label']
            missing = [col for col in required_cols if col not in raw_df.columns]
            if missing:
                raise KeyError(f"Missing columns in WESAD data: {missing}")

            # Directly use ACC columns from raw data
            clean_df = pd.DataFrame({
                'bvp': raw_df['bvp'],
                'acc_x': raw_df['acc_x'],
                'acc_y': raw_df['acc_y'],
                'acc_z': raw_df['acc_z'],
                'label': raw_df['label'],
                'subject_id': subject_id,
                'skin_tone': 'clean',
                'device': 'clean'
            })
            
            # Generate proper timestamps with explicit freq
            freq = pd.Timedelta(milliseconds=33.333)
            clean_df.index = pd.date_range(
                start=pd.Timestamp.now().floor('D'), 
                periods=len(clean_df), 
                freq=freq,
                name='timestamp'
            )
            
            # Add validation
            if not self._validate_sample_rate(clean_df):
                raise ValueError("Invalid sampling rate in clean data")
            
            # Add ACC clipping
            clean_df[self.ACC_COLS] = clean_df[self.ACC_COLS].clip(-3.5, 3.5)
            
            # Remap sensors
            sensor_df = self.remap_sensors(clean_df)
            
            # Map labels to unified scheme
            sensor_df['label'] = self.map_labels(sensor_df['label'], 'wesad')
            
            # Add dataset identifier
            sensor_df['dataset'] = 'wesad'
            sensor_df['device'] = 'clean'
            sensor_df['skin_tone'] = 'none'  # Add default for clean data

            if 'noise_level' not in sensor_df.columns:
                sensor_df['noise_level'] = 0.0
            
            # Validate
            if not self.validate_output(sensor_df):
                raise ValueError("Validation failed")
                
            # Save clean data
            clean_path = self.output_dir/f"clean_wesad_s{subject_id}.parquet"
            sensor_df.to_parquet(clean_path, index=True)

            # Add device noise variants
            noisy_sim = NoiseSimulator()
            skin_tones = ['I-II', 'III-IV', 'V-VI']
            
            files = [clean_path]  # Start with clean path
            
            for device in ['apple_watch', 'galaxy_watch']:
                for skin in skin_tones:
                    noisy_df = noisy_sim.add_device_noise(
                        sensor_df.copy(), 
                        device=device,
                        skin_tone=skin  # Use skin parameter
                    )
                    noisy_df['skin_tone'] = skin  # Track in metadata
                    
                    # Update filename to include skin
                    noisy_path = self.output_dir / f"{device}_{skin}_wesad_s{subject_id}.parquet"
                    noisy_df.to_parquet(noisy_path, index=True)
                    files.append(noisy_path)  # Collect all generated files

            return {
                'status': 'success', 
                'subject': subject_id,
                'files': [str(p) for p in files]  # Include all file paths
            }
            
        except Exception as e:
            self.logger.error(f"Failed processing subject {subject_id}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'subject': subject_id,
                'files': []  # Maintain consistent structure
            }

    def _validate_subject(self, data: pd.DataFrame) -> bool:
        """WESAD-specific validation"""
        # Check BVP range specific to WESAD
        bvp_valid = data['bvp'].between(-2.5, 2.5).mean() > 0.99
        if not bvp_valid:
            self.logger.warning("BVP values outside WESAD expected range")
            
        # Check label distribution
        label_counts = data['label'].value_counts(normalize=True)
        if label_counts.get(3, 0) < 0.05:  # At least 5% meditation samples
            self.logger.warning("Low meditation class samples")
            
        return super().validate_output(data) and bvp_valid 

    def load_subject(self, subject_id: int) -> pd.DataFrame:
        """Implementation required by BaseDataLoader"""
        return self.loader.load_subject(subject_id)

    def get_labels(self) -> Dict[str, Any]:
        """Implementation required by BaseDataLoader"""
        return {
            'original_labels': self.loader.get_labels(),
            'mapped_labels': self.TARGET_LABELS
        } 