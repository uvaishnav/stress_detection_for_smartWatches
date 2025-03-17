import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

from .base_feature_extractor import BaseFeatureExtractor

class ContextAwareExtractor(BaseFeatureExtractor):
    """
    Extracts context-aware features that incorporate motion data and device-specific information.
    
    This extractor focuses on features that consider the context in which the physiological
    signals were recorded, such as motion artifacts, device positioning, and environmental factors.
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30, device_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the context-aware feature extractor.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
            device_info: Dictionary containing device-specific information (default: None)
        """
        super().__init__(window_size, overlap)
        self.sampling_rate = sampling_rate
        self.device_info = device_info or {}
        
    def extract_features(self, window: np.ndarray, 
                         acc_x: Optional[np.ndarray] = None,
                         acc_y: Optional[np.ndarray] = None,
                         acc_z: Optional[np.ndarray] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extract context-aware features from a single window.
        
        Args:
            window: Numpy array containing the PPG signal window
            acc_x: Numpy array containing the accelerometer X-axis data (optional)
            acc_y: Numpy array containing the accelerometer Y-axis data (optional)
            acc_z: Numpy array containing the accelerometer Z-axis data (optional)
            metadata: Dictionary containing additional metadata for the window (optional)
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        features = {}
        
        # Extract motion-related features if accelerometer data is available
        if acc_x is not None and acc_y is not None and acc_z is not None:
            features.update(self._extract_motion_features(acc_x, acc_y, acc_z))
            features.update(self._extract_motion_ppg_correlation(window, acc_x, acc_y, acc_z))
        
        # Extract device-specific features if device info is available
        if self.device_info:
            features.update(self._extract_device_features(window))
        
        # Extract metadata features if metadata is available
        if metadata:
            features.update(self._extract_metadata_features(metadata))
        
        return features
    
    def _extract_motion_features(self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Dict[str, float]:
        """
        Extract features from accelerometer data.
        
        Args:
            acc_x: Numpy array containing the accelerometer X-axis data
            acc_y: Numpy array containing the accelerometer Y-axis data
            acc_z: Numpy array containing the accelerometer Z-axis data
            
        Returns:
            Dictionary of motion-related features
        """
        features = {}
        
        # Ensure all arrays have the same length
        min_length = min(len(acc_x), len(acc_y), len(acc_z))
        acc_x = acc_x[:min_length]
        acc_y = acc_y[:min_length]
        acc_z = acc_z[:min_length]
        
        # Calculate magnitude of acceleration
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Basic statistical features for each axis
        for axis, data in zip(['x', 'y', 'z'], [acc_x, acc_y, acc_z]):
            features[f'acc_{axis}_mean'] = np.mean(data)
            features[f'acc_{axis}_std'] = np.std(data)
            features[f'acc_{axis}_min'] = np.min(data)
            features[f'acc_{axis}_max'] = np.max(data)
            features[f'acc_{axis}_range'] = np.max(data) - np.min(data)
            features[f'acc_{axis}_median'] = np.median(data)
            features[f'acc_{axis}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
            features[f'acc_{axis}_energy'] = np.sum(data**2) / len(data)
            
            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            features[f'acc_{axis}_zero_crossings'] = len(zero_crossings)
            
            # Peaks
            peaks, _ = self._find_peaks(data)
            features[f'acc_{axis}_peak_count'] = len(peaks)
        
        # Features for acceleration magnitude
        features['acc_mag_mean'] = np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_mag)
        features['acc_mag_min'] = np.min(acc_mag)
        features['acc_mag_max'] = np.max(acc_mag)
        features['acc_mag_range'] = np.max(acc_mag) - np.min(acc_mag)
        
        # Activity level estimation
        features['activity_level'] = np.mean(acc_mag)
        features['activity_variability'] = np.std(acc_mag)
        
        # Jerk (derivative of acceleration)
        jerk_x = np.diff(acc_x) * self.sampling_rate
        jerk_y = np.diff(acc_y) * self.sampling_rate
        jerk_z = np.diff(acc_z) * self.sampling_rate
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        features['jerk_mean'] = np.mean(jerk_mag)
        features['jerk_std'] = np.std(jerk_mag)
        features['jerk_max'] = np.max(jerk_mag)
        
        # Dominant frequency
        if len(acc_mag) > 10:  # Ensure enough data points for FFT
            fft_mag = np.abs(np.fft.rfft(acc_mag))
            freqs = np.fft.rfftfreq(len(acc_mag), 1/self.sampling_rate)
            
            # Exclude DC component (0 Hz)
            fft_mag = fft_mag[1:]
            freqs = freqs[1:]
            
            if len(freqs) > 0:
                dominant_idx = np.argmax(fft_mag)
                features['acc_dominant_freq'] = freqs[dominant_idx]
                features['acc_dominant_power'] = fft_mag[dominant_idx]
        
        return features
    
    def _extract_motion_ppg_correlation(self, ppg: np.ndarray, 
                                       acc_x: np.ndarray, 
                                       acc_y: np.ndarray, 
                                       acc_z: np.ndarray) -> Dict[str, float]:
        """
        Extract features related to the correlation between PPG and motion.
        
        Args:
            ppg: Numpy array containing the PPG signal window
            acc_x: Numpy array containing the accelerometer X-axis data
            acc_y: Numpy array containing the accelerometer Y-axis data
            acc_z: Numpy array containing the accelerometer Z-axis data
            
        Returns:
            Dictionary of correlation features
        """
        features = {}
        
        # Ensure all arrays have the same length
        min_length = min(len(ppg), len(acc_x), len(acc_y), len(acc_z))
        ppg = ppg[:min_length]
        acc_x = acc_x[:min_length]
        acc_y = acc_y[:min_length]
        acc_z = acc_z[:min_length]
        
        # Calculate magnitude of acceleration
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Correlation between PPG and each acceleration axis
        for axis, data in zip(['x', 'y', 'z', 'mag'], [acc_x, acc_y, acc_z, acc_mag]):
            corr, p_value = stats.pearsonr(ppg, data)
            features[f'ppg_acc_{axis}_corr'] = corr
            features[f'ppg_acc_{axis}_corr_p'] = p_value
        
        # Motion artifact probability estimation
        # Higher correlation with motion typically indicates more motion artifacts
        features['motion_artifact_prob'] = np.abs(features['ppg_acc_mag_corr'])
        
        # Signal quality estimation (inverse of motion artifact probability)
        features['signal_quality'] = 1.0 - features['motion_artifact_prob']
        
        # Cross-correlation at different lags
        max_lag = min(20, min_length // 4)  # Maximum lag in samples
        
        for axis, data in zip(['x', 'y', 'z'], [acc_x, acc_y, acc_z]):
            xcorr = np.correlate(ppg, data, mode='full')
            xcorr = xcorr[len(xcorr)//2 - max_lag:len(xcorr)//2 + max_lag + 1]
            
            # Maximum cross-correlation and its lag
            max_xcorr_idx = np.argmax(np.abs(xcorr))
            max_xcorr = xcorr[max_xcorr_idx]
            lag = max_xcorr_idx - max_lag
            
            features[f'ppg_acc_{axis}_max_xcorr'] = max_xcorr
            features[f'ppg_acc_{axis}_max_xcorr_lag'] = lag / self.sampling_rate  # Convert to seconds
        
        return features
    
    def _extract_device_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract features that incorporate device-specific information.
        
        Args:
            window: Numpy array containing the PPG signal window
            
        Returns:
            Dictionary of device-specific features
        """
        features = {}
        
        # Device type normalization factor
        if 'device_type' in self.device_info:
            device_type = self.device_info['device_type']
            
            # Different devices might have different baseline values or scaling factors
            if device_type == 'apple_watch':
                features['device_norm_factor'] = 1.0
            elif device_type == 'fitbit':
                features['device_norm_factor'] = 0.85
            elif device_type == 'samsung_galaxy':
                features['device_norm_factor'] = 0.92
            else:
                features['device_norm_factor'] = 1.0
            
            # Apply device-specific normalization
            features['ppg_mean_normalized'] = np.mean(window) * features['device_norm_factor']
            features['ppg_std_normalized'] = np.std(window) * features['device_norm_factor']
        
        # Sensor quality factor
        if 'sensor_quality' in self.device_info:
            features['sensor_quality_factor'] = float(self.device_info['sensor_quality'])
        
        # Device position adjustment
        if 'wearing_position' in self.device_info:
            position = self.device_info['wearing_position']
            
            if position == 'wrist_top':
                features['position_factor'] = 1.0
            elif position == 'wrist_bottom':
                features['position_factor'] = 0.9
            elif position == 'loose':
                features['position_factor'] = 0.7
            else:
                features['position_factor'] = 1.0
        
        return features
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from additional metadata.
        
        Args:
            metadata: Dictionary containing additional metadata for the window
            
        Returns:
            Dictionary of metadata-based features
        """
        features = {}
        
        # Time of day factor (circadian rhythm effects)
        if 'time_of_day' in metadata:
            hour = metadata['time_of_day']
            # Normalize to [0, 1] with peak at noon
            time_factor = 1.0 - abs(hour - 12) / 12
            features['time_of_day_factor'] = time_factor
        
        # User demographic factors
        if 'age' in metadata:
            age = metadata['age']
            # Age can affect baseline HRV
            if age < 30:
                features['age_factor'] = 1.2
            elif age < 50:
                features['age_factor'] = 1.0
            else:
                features['age_factor'] = 0.8
        
        if 'gender' in metadata:
            gender = metadata['gender']
            # Gender can affect baseline HRV
            if gender == 'female':
                features['gender_factor'] = 1.1
            else:
                features['gender_factor'] = 1.0
        
        # Environmental factors
        if 'temperature' in metadata:
            temp = metadata['temperature']
            # Temperature can affect skin conductance and blood flow
            features['temperature_factor'] = 1.0 + (temp - 20) / 100
        
        return features
    
    def _find_peaks(self, signal: np.ndarray, height: Optional[float] = None, 
                   distance: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in a signal.
        
        Args:
            signal: Numpy array containing the signal
            height: Minimum height of peaks (default: None)
            distance: Minimum distance between peaks in samples (default: None)
            
        Returns:
            Tuple of (peak indices, peak heights)
        """
        if distance is None:
            distance = max(1, int(0.25 * self.sampling_rate))
        
        # Find peaks
        peak_indices = []
        peak_heights = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if height is None or signal[i] > height:
                    # Check distance from last peak
                    if not peak_indices or i - peak_indices[-1] >= distance:
                        peak_indices.append(i)
                        peak_heights.append(signal[i])
        
        return np.array(peak_indices), np.array(peak_heights)
    
    def process_dataframe_with_context(self, df: pd.DataFrame, ppg_col: str, 
                                      acc_x_col: Optional[str] = None,
                                      acc_y_col: Optional[str] = None,
                                      acc_z_col: Optional[str] = None,
                                      metadata_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process a DataFrame with context information.
        
        Args:
            df: DataFrame containing the data
            ppg_col: Name of the column containing PPG data
            acc_x_col: Name of the column containing accelerometer X-axis data (optional)
            acc_y_col: Name of the column containing accelerometer Y-axis data (optional)
            acc_z_col: Name of the column containing accelerometer Z-axis data (optional)
            metadata_cols: List of column names containing metadata (optional)
            
        Returns:
            DataFrame containing extracted features
        """
        # Create windows from PPG data
        windows, indices = self.create_windows_from_df(df, ppg_col)
        
        # Initialize list to store feature dictionaries
        feature_dicts = []
        
        for i, (window, start_idx) in enumerate(zip(windows, indices)):
            # Extract PPG window
            ppg_window = window
            
            # Extract accelerometer windows if available
            acc_x_window = None
            acc_y_window = None
            acc_z_window = None
            
            if acc_x_col and acc_y_col and acc_z_col:
                end_idx = start_idx + self.window_size
                if end_idx <= len(df):
                    acc_x_window = df[acc_x_col].values[start_idx:end_idx]
                    acc_y_window = df[acc_y_col].values[start_idx:end_idx]
                    acc_z_window = df[acc_z_col].values[start_idx:end_idx]
            
            # Extract metadata if available
            metadata = {}
            if metadata_cols:
                for col in metadata_cols:
                    if col in df.columns:
                        # Use the value at the start of the window
                        metadata[col] = df[col].iloc[start_idx]
            
            # Extract features
            features = self.extract_features(
                ppg_window, 
                acc_x=acc_x_window,
                acc_y=acc_y_window,
                acc_z=acc_z_window,
                metadata=metadata
            )
            
            # Add window index
            features['window_idx'] = i
            features['start_idx'] = start_idx
            
            feature_dicts.append(features)
        
        # Convert list of dictionaries to DataFrame
        features_df = pd.DataFrame(feature_dicts)
        
        return features_df 