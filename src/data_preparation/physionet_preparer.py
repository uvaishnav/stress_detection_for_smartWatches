import pandas as pd
from pathlib import Path
from typing import Dict, Any
from data_loading.physionet_loader import PhysioNetLoader
from .base_preparer import BasePreparer
import logging
from data_preparation.noise_simulator import NoiseSimulator
import numpy as np

class PhysioNetPreparer(BasePreparer):
    """Handles PhysioNet-specific preparation tasks"""
    
    # PhysioNet-specific constants
    ACC_REMAP = {
        'acc_x': 'acc_y',  # North → East
        'acc_y': 'acc_x',  # East → North
        'acc_z': 'acc_z'   # Down → Up (invert later)
    }
    
    def __init__(self, data_path: str, output_dir: str = "data/processed/physionet"):
        super().__init__(data_path, output_dir)
        self.loader = PhysioNetLoader(data_path)
        self.logger = logging.getLogger('PhysioNetPreparer')

    def remap_sensors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert NED to ENU orientation"""
        # PhysioNet ACC uses NED orientation (North-East-Down)
        # Remap to ENU (East-North-Up) like WESAD
        
        # Step 1: Rename axes
        remapped = data.rename(columns=self.ACC_REMAP)
        
        # Step 2: Invert Z-axis (Down → Up)
        remapped['acc_z'] *= -1
        
        # Step 3: Validate
        if not {'acc_x', 'acc_y', 'acc_z'}.issubset(remapped.columns):
            self.logger.error("ACC remapping failed")
            raise ValueError("Invalid ACC columns after remapping")
            
        return remapped

    def _add_physionet_timestamps(self, df: pd.DataFrame, subject_id: int) -> pd.DataFrame:
        """Precise timestamp generation with length validation"""
        try:
            start_time = self.loader.get_recording_start(subject_id)
            freq = pd.Timedelta(microseconds=33333)  # Exact 30Hz (33.333ms)
            new_index = pd.date_range(
                start=start_time,
                periods=len(df),
                freq=freq,
                tz='UTC'
            )
            df.index = new_index
        except Exception as e:
            self.logger.error(f"Timestamp generation failed: {str(e)}")
            raise ValueError("Could not generate valid timestamps") from e
        
        # Validate index length
        if len(df.index) != len(df):
            raise ValueError(f"Timestamp count mismatch ({len(df.index)} vs {len(df)})")
        
        return df

    def _unify_sampling_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resampling with index preservation"""
        # Create reference index
        original_length = len(df)
        target_freq = '33.333333ms'  # Exact 30Hz
        
        # Generate new index
        new_index = pd.date_range(
            start=df.index[0],
            periods=original_length,
            freq=target_freq,
            tz='UTC'
        )
        
        # Reindex without changing data
        return df.reindex(new_index, method='nearest').ffill()

    def process_subject(self, subject_id: int, exam_type: str) -> Dict:
        """Main processing with tags validation"""
        if self.loader.is_tags_empty(subject_id, exam_type):
            return {
                'status': 'skipped', 
                'error': f'Empty tags for {exam_type}'
            }
        try:
            # Set numeric subject ID for schema compliance
            self.current_subject = int(subject_id)  # Convert to pure integer
            
            raw_df = self.loader.load_subject(subject_id, exam_type)
            
            # Keep original lowercase column names
            clean_df = pd.DataFrame({
                'hr': raw_df['hr'],
                'eda': raw_df['eda'],
                'temp': raw_df['temp'],
                'bvp': raw_df['bvp'],  # Keep lowercase
                'acc_x': raw_df['acc_x'],
                'acc_y': raw_df['acc_y'], 
                'acc_z': raw_df['acc_z'],
                'label': raw_df['event'],
                'subject_id': subject_id,
                'exam_type': exam_type,
            })
            
            target_length = len(raw_df)  # Capture original length
            
            clean_df = self._add_physionet_timestamps(clean_df, subject_id)
            self.logger.debug(f"Clean data index start: {clean_df.index[0]}, length: {len(clean_df)}")
            
            unified_df = self._unify_sampling_rate(clean_df)
            
            
            # Add required metadata
            unified_df['dataset'] = 'physionet'
            unified_df['device'] = 'clean'
            unified_df['skin_tone'] = 'none'  # Add default for clean data
            
            # Add noise level tracking
            if 'noise_level' not in unified_df.columns:
                unified_df['noise_level'] = 0.0
            
            # Process data
            sensor_df = self.remap_sensors(unified_df)
            
            # Map labels (0=baseline, 1=stress)
            sensor_df['label'] = self.map_labels(sensor_df['label'], 'physionet')
            
            # Validate
            if not self.validate_output(sensor_df):
                raise ValueError("Validation failed")
                
            # Add sampling rate validation
            if not self._validate_sample_rate(unified_df):
                raise ValueError("Invalid sampling rate after processing")
                
            # Save clean data
            clean_path = self._save_processed(unified_df, subject_id, exam_type, 'clean')
            
            # Add device noise variants
            noisy_sim = NoiseSimulator()
            device_data = {}
            
            # Add skin tone loop to device variants
            skin_tones = ['I-II', 'III-IV', 'V-VI']
            
            for device in ['apple_watch', 'galaxy_watch']:
                for skin in skin_tones:
                    noisy_df = noisy_sim.add_device_noise(
                        sensor_df.copy(),
                        device=device,
                        skin_tone=skin
                    )
                    # Explicitly set skin tone column
                    noisy_df['skin_tone'] = skin  
                    
                    # Include skin in filename
                    path = self._save_processed(noisy_df, subject_id, exam_type, f"{device}_{skin}")
                    device_data[f"{device}_{skin}"] = noisy_df
            
            # Validate outputs (only check noisy files)
            self._validate_outputs(unified_df, list(device_data.values()))
            
            # Generate and save all data versions
            files = [str(clean_path)]  # Start with clean path
            for device, data in device_data.items():
                path = self._save_processed(data, subject_id, exam_type, device)
                files.append(str(path))
            
            return {
                'status': 'success',
                'files': files,
                'subject': subject_id,
                'exam': exam_type
            }
            
        except Exception as e:
            self.logger.error(f"Failed {exam_type} for subject {subject_id}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'subject': subject_id,
                'exam': exam_type,
                'files': []
            }

    def _save_processed(self, df: pd.DataFrame, subject_id: int, exam_type: str, device: str) -> Path:
        # Preserve index with exact timezone and precision
        df = df.copy()
        df.index = df.index.tz_convert('UTC').round('1ms')  # Force UTC and millisecond precision
        path = self.output_dir / f"{device}_physionet_s{subject_id}_{exam_type}.parquet"
        df.to_parquet(path, index=True, coerce_timestamps='ms', allow_truncated_timestamps=False)
        return path

    def _validate_outputs(self, clean_df: pd.DataFrame, data_list: list):
        """Validate a list of DataFrames against clean reference"""
        ref_values = clean_df.index.tz_convert(None).astype('datetime64[ns]').values
        
        for df in data_list:
            current_values = df.index.tz_convert(None).astype('datetime64[ns]').values
            if not np.array_equal(ref_values, current_values):
                raise ValueError("Index mismatch in generated data")

    def _validate_subject(self, data: pd.DataFrame) -> bool:
        """PhysioNet-specific checks"""
        # Check label ratio (expect sparse events)
        label_ratio = data['label'].mean()
        if label_ratio > 0.3:
            self.logger.warning(f"High stress ratio: {label_ratio:.1%}")
            
        # Check ACC range after remapping
        acc_valid = data[['acc_x', 'acc_y', 'acc_z']].abs().max().max() <= 3.5
        if not acc_valid:
            self.logger.warning("ACC exceeds 3.5g after remapping")
            
        return super().validate_output(data) and acc_valid 

    # Implement abstract methods
    def load_subject(self, subject_id: int, exam_type: str) -> pd.DataFrame:
        """Load raw subject data"""
        return self.loader.load_subject(subject_id, exam_type)

    def get_labels(self) -> Dict[str, Any]:
        """Get label mapping documentation"""
        return {
            'original_labels': self.loader.get_labels(),
            'mapped_labels': self.TARGET_LABELS
        } 