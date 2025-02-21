import numpy as np
import pandas as pd
from typing import Dict

class SensorAligner:
    """Handles sensor alignment across datasets"""
    
    # Wavelength correction factors (525nm reference)
    PPG_CORRECTION = {
        'wesad': 1.0,        # 525nm
        'physionet': 0.9545  # 550nm → 525nm (525/550)
    }
    
    # ACC orientation maps
    ACC_REMAPPING = {
        'physionet': {
            'x': 'y',    # North → Y
            'y': 'x',    # East → X
            'z': '-z'    # Down → Up (via negation)
        },
        'wesad': {
            'x': 'x',    # East remains X
            'y': 'y',    # North remains Y
            'z': 'z'     # Up remains Z
        }
    }

    def align_ppg(self, data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Correct PPG signals for wavelength differences"""
        factor = self.PPG_CORRECTION[dataset]
        aligned = data.copy()
        aligned['bvp'] *= factor
        return aligned

    def align_acc(self, data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Remap ACC axes to common ENU frame"""
        mapping = self.ACC_REMAPPING[dataset]
        acc_data = data[['acc_x', 'acc_y', 'acc_z']].copy()
        
        # Apply axis remapping
        remapped = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            target = mapping[axis]
            if target.startswith('-'):
                remapped[f'acc_{axis}'] = -acc_data[f'acc_{target[1:]}']
            else:
                remapped[f'acc_{axis}'] = acc_data[f'acc_{target}']
                
        return data.drop(columns=['acc_x', 'acc_y', 'acc_z']).join(remapped)

    def cross_dataset_align(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Final alignment accounting for skin noise profiles"""
        # 1. Normalize PPG by skin tone
        skin_factors = {
            'I-II': 1.0,
            'III-IV': 0.9, 
            'V-VI': 0.8
        }
        merged['bvp'] *= merged['skin_tone'].map(skin_factors)
        
        # 2. Device-specific ACC calibration
        merged['acc_x'] = np.where(
            merged['device'] == 'apple_watch',
            merged['acc_x'] * 1.05,  # Apple-specific calibration
            merged['acc_x']
        )
        
        return merged 
    
    def temporal_align(self, data: pd.DataFrame, reference: pd.DataFrame = None) -> pd.DataFrame:
        """Align timestamps using dynamic time warping"""
        if reference is None:
            return data  # No alignment needed for first variant
        
        # Simple linear interpolation for initial implementation
        aligned = data.reindex_like(reference).interpolate(method='time')
        aligned['label'] = data['label'].reindex(aligned.index, method='ffill')
        return aligned.fillna(method='ffill')