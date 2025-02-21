import numpy as np
import pandas as pd
from typing import Dict

class NoiseSimulator:
    """Simulates smartwatch-specific noise characteristics"""
    
    DEVICE_PROFILES = {
        'apple_watch': {
            'bvp_noise': 0.12,
            'acc_noise': 0.08,
            'thermal_drift': 0.0002
        },
        'galaxy_watch': {
            'bvp_noise': 0.18,
            'acc_noise': 0.12,
            'thermal_drift': 0.0005  
        }
    }

    def __init__(self):
        # Revised parameters (absolute μV)
        self.skin_effects = {
            'I-II': {
                'gain': 1.0,  # Signal multiplier
                'noise': 4.0,  # μV RMS
                'motion_multiplier': 0.8
            },
            'III-IV': {
                'gain': 0.8,
                'noise': 6.0,
                'motion_multiplier': 1.0
            },
            'V-VI': {
                'gain': 0.6,
                'noise': 8.0,
                'motion_multiplier': 1.2
            }
        }

    def add_device_noise(self, data: pd.DataFrame, device: str, skin_tone: str = 'III-IV') -> pd.DataFrame:
        """Absolute length preservation"""
        original_length = len(data)
        original_index = data.index
        
        # Preserve original BVP values
        bvp_backup = data['bvp'].copy()
        
        # Apply noise
        noisy_df = self._apply_ppg_noise(data, device, skin_tone)
        noisy_df = self._add_motion_effects(noisy_df, device, skin_tone)
        
        # Force original dimensions
        noisy_df = noisy_df.iloc[:original_length]  # Trim
        noisy_df.index = original_index  # Align
        
        # Restore original values where noise introduced NaNs
        noisy_df['bvp'] = noisy_df['bvp'].fillna(bvp_backup)

        noisy_df['device']=device
        
        # Forward fill any remaining nulls
        noisy_df.ffill(inplace=True)
        
        return noisy_df

    def _apply_ppg_noise(self, data: pd.DataFrame, device: str, skin_tone: str) -> pd.DataFrame:
        # Preserve original timestamps
        original_index = data.index.copy()
        
        # Initialize noise level
        if 'noise_level' not in data.columns:
            data['noise_level'] = 0.0
        
        # Calculate dynamic components
        params = self.skin_effects[skin_tone]
        ppg_contribution = params['gain'] * self.DEVICE_PROFILES[device]['bvp_noise']
        motion_contribution = 0.2 * params['motion_multiplier'] * self.DEVICE_PROFILES[device]['acc_noise']
        
        # Update noise level incrementally
        data['noise_level'] += (0.6 * ppg_contribution) + (0.4 * motion_contribution)
        
        noisy_data = data.copy()
        n_samples = len(noisy_data)
        
        # Base device noise
        if device == 'apple_watch':
            dc_offset = noisy_data['bvp'].mean()
            ac_component = noisy_data['bvp'] - dc_offset
            
            # 1. Apply gain to AC component
            ac_component = ac_component * params['gain']
            
            # 2. Add skin-specific noise (μV scale)
            ac_component += np.random.normal(0, params['noise'], n_samples)
            
            # 3. Reconstruct signal with base noise
            noisy_data['bvp'] = dc_offset + ac_component
            
            # 4. Add motion artifacts ONLY to accelerometer
            motion_scale = 0.2 * params['motion_multiplier']
            for axis in ['acc_x', 'acc_y', 'acc_z']:
                noisy_data[axis] += np.random.normal(0, motion_scale, n_samples)
            
        elif device == 'galaxy_watch':
            # Galaxy Watch BVP noise (low-frequency drift)
            drift = np.linspace(0, 0.2, n_samples)
            noisy_data['bvp'] += drift
            
            # ACC amplification on X-axis
            noisy_data['acc_x'] *= 1.1
            
            # Different skin response for Samsung's green LED
            noisy_data['bvp'] *= params['gain'] * 0.9  
            noisy_data['bvp'] += np.random.laplace(0, params['noise'], n_samples)
            
        else:
            raise ValueError(f"Unknown device: {device}")
            
        # Keep original timestamps
        noisy_data.index = original_index
        return noisy_data

    def _frequency_mask(self, index: pd.DatetimeIndex, cutoff: float) -> np.ndarray:
        """Create frequency-dependent noise mask"""
        dt = (index[1] - index[0]).total_seconds()
        f = np.fft.rfftfreq(len(index), dt)
        mask = np.where(f < cutoff, 1.0, 0.5)
        return np.fft.irfft(mask * np.random.randn(len(mask))).real[:len(index)]

    def add_motion_artifacts(self, data: pd.DataFrame, intensity: float = 0.3) -> pd.DataFrame:
        """Simulate motion artifacts based on accelerometer data"""
        noisy = data.copy()
        
        # 1. Detect movement periods
        acc_norm = np.linalg.norm(data[['acc_x','acc_y','acc_z']], axis=1)
        movement = (acc_norm > 1.2).astype(float)  # > 1.2g threshold
        
        # 2. Add transient spikes during movement
        spike_prob = movement * intensity
        spikes = np.random.binomial(1, spike_prob) * np.random.normal(0, 50, len(data))
        noisy['bvp'] += spikes
        
        # 3. Baseline wander
        wander = np.convolve(movement, np.ones(100)/100, mode='same') 
        noisy['bvp'] += wander * 20
        
        return noisy 

    def _add_motion_effects(self, data: pd.DataFrame, device: str, skin_tone: str) -> pd.DataFrame:
        """Apply motion artifacts to ACC and BVP signals"""
        # Preserve existing motion artifact logic
        motion_params = self.DEVICE_PROFILES[device]
        skin_params = self.skin_effects[skin_tone]
        
        # Existing acceleration-based noise implementation
        data['bvp'] += self._generate_motion_noise(data['acc_x'], data['acc_y'], data['acc_z'])
        return data 

    def _generate_motion_noise(self, acc_x: pd.Series, acc_y: pd.Series, acc_z: pd.Series) -> pd.Series:
        """Use existing acc_noise parameter for motion artifacts"""
        # Get parameters from original device profile
        device_params = self.DEVICE_PROFILES.get('apple_watch', {})
        skin_params = self.skin_effects.get('medium', {})  # Default to medium if not specified
        
        motion_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Use acc_noise parameter for sensitivity
        motion_noise = motion_mag * device_params['acc_noise']
        
        # Add baseline from bvp_noise parameter
        motion_noise += np.random.normal(
            scale=device_params.get('bvp_noise', 0.12),
            size=len(motion_noise)
        )
        
        return motion_noise.values 