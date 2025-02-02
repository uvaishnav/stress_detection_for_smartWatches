from pathlib import Path
import numpy as np
import pandas as pd
from .base_loader import BaseDataLoader
from typing import Dict, Any

class PhysioNetLoader(BaseDataLoader):
    """Loader for PhysioNet Exam Stress dataset"""
    
    # Exam period configuration
    EXAM_DURATIONS = {
        'Midterm 1': 5400,  # 1.5h in seconds
        'Midterm 2': 5400,
        'Final': 10800      # 3h
    }
    
    # Expected sensor rates (Hz)
    SENSOR_RATES = {
        'bvp': 64,
        'acc': 32,
        'eda': 4,
        'temp': 4,
        'hr': 1
    }

    # Constants at top of file
    EXPECTED_COLUMNS = [
        'bvp', 'acc_x', 'acc_y', 'acc_z', 
        'eda', 'temp', 'hr', 'event',
        'subject_id', 'session', 'sampling_rate'
    ]

    def __init__(self, data_path: str, target_rate: int = 30):
        super().__init__(data_path, target_rate, dataset_name='physionet')
        self.raw_sample_rate = 64  # BVP @ 64Hz
        self.subject_id = None

    def _load_and_convert(self, session_path: Path, sensor: str) -> pd.DataFrame:
        """Robust loading with explicit column validation"""
        try:
            file_path = session_path / f"{sensor.upper()}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"{sensor.upper()}.csv not found")
            
            t, sr, data = self._load_sensor_data(file_path)
            df = self._create_timeseries(data, t, sr, sensor)
            
            # Validate expected columns
            if sensor == 'acc' and not {'acc_x','acc_y','acc_z'}.issubset(df.columns):
                raise ValueError("Missing ACC axes")
            elif sensor != 'acc' and sensor not in df.columns:
                raise ValueError(f"Missing {sensor} data column")
            
            return df
        
        except Exception as e:
            self.logger.error(f"{sensor.upper()} load failed: {str(e)}")
            return pd.DataFrame(columns=self.EXPECTED_COLUMNS)

    def _load_events(self, session_path: Path) -> pd.Series:
        """Load event markers from tags.csv"""
        try:
            tags_file = session_path / 'tags.csv'
            if not tags_file.exists():
                return pd.Series(name='event', dtype='float32')
            
            # Read event timestamps
            events = pd.read_csv(
                tags_file,
                header=None,
                names=['timestamp'],
                dtype=np.float32
            )
            
            # Convert to datetime
            events['datetime'] = pd.to_datetime(
                events['timestamp'], 
                unit='s', 
                utc=True
            )
            return events.set_index('datetime')['timestamp'].notnull().astype(int)
        
        except Exception as e:
            self.logger.error(f"Failed to load events: {str(e)}")
            return pd.Series(name='event', dtype='float32')

    def load_subject(self, subject_id: int, session: str) -> pd.DataFrame:
        """Load and align data with event periods from tags.csv"""
        try:
            session_path = self.data_path / f"S{subject_id}" / session
            target_freq = f"{1000//self.target_rate}ms"  # 33ms for 30Hz
            
            # 1. Create empty resampled index for entire session duration
            first_sensor = next((s for s in ['ACC', 'BVP'] if (session_path/f"{s}.csv").exists()), None)
            if not first_sensor:
                return pd.DataFrame(columns=self.EXPECTED_COLUMNS)
            
            _, _, sample_data = self._load_sensor_data(session_path/f"{first_sensor}.csv")
            full_index = sample_data.resample(target_freq).asfreq().index
            base_df = pd.DataFrame(index=full_index)

            # 2. Load and resample all sensors
            sensor_data = {}
            for sensor in ['ACC', 'BVP', 'EDA', 'TEMP', 'HR']:
                path = session_path/f"{sensor}.csv"
                if path.exists():
                    _, sr, data = self._load_sensor_data(path)
                    resampled = data.resample(target_freq).mean()
                    sensor_data[sensor.lower()] = resampled.reindex(full_index)

            # 3. Merge all sensor data
            for name, data in sensor_data.items():
                if name == 'acc':
                    base_df[['acc_x', 'acc_y', 'acc_z']] = data[['acc_x', 'acc_y', 'acc_z']]
                else:
                    base_df[name] = data[name] if name in data.columns else 0.0

            # 4. Load and apply event periods from tags.csv
            base_df['event'] = 0
            tags_file = session_path / 'tags.csv'
            if tags_file.exists() and tags_file.stat().st_size > 0:
                try:
                    tags = pd.read_csv(tags_file, header=None, names=['start', 'end'])
                    for _, row in tags.iterrows():
                        start = pd.to_datetime(row['start'], unit='s', utc=True)
                        end = pd.to_datetime(row['end'], unit='s', utc=True)
                        base_df.loc[start:end, 'event'] = 1
                except Exception as e:
                    self.logger.error(f"Tags processing failed: {str(e)}")

            # 5. Add metadata and final cleanup
            base_df['subject_id'] = subject_id
            base_df['session'] = session
            base_df['sampling_rate'] = self.target_rate
            base_df.ffill(inplace=True)  # Handle any resampling gaps
            
            return base_df.reindex(columns=self.EXPECTED_COLUMNS, fill_value=0.0)

        except Exception as e:
            self.logger.error(f"Subject load failed: {str(e)}")
            return pd.DataFrame(columns=self.EXPECTED_COLUMNS)

    def _load_and_process_sensor(self, session_path: Path, sensor: str, native_rate: float) -> tuple:
        """Helper to load and validate individual sensors"""
        try:
            file_path = session_path / f"{sensor.upper()}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"{sensor.upper()} file missing")
            
            t, sr, data = self._load_sensor_data(file_path)
            df = self._create_timeseries(data, t, sr, sensor)
            
            # Validate sensor-specific columns
            if sensor == 'acc' and not {'acc_x', 'acc_y', 'acc_z'}.issubset(df.columns):
                raise ValueError("Invalid ACC data format")
            
            return (t, native_rate, df)  # Return native rate for resampling
        
        except Exception as e:
            self.logger.warning(f"Skipping {sensor}: {str(e)}")
            return (0, 0, pd.DataFrame())

    def _load_sensor_data(self, file_path: Path) -> tuple[float, float, pd.DataFrame]:
        """Load sensor data with proper datetime index creation"""
        try:
            with open(file_path, 'r') as f:
                # Read header lines
                if 'ACC' in file_path.name:
                    initial_time = float(f.readline().split(',')[0].strip())
                    sample_rate = float(f.readline().split(',')[0].strip())
                else:
                    initial_time = float(f.readline().strip())
                    sample_rate = float(f.readline().strip())                
                # Load sensor data
                if 'ACC' in file_path.name:
                    df = pd.read_csv(
                        f, 
                        header=None,
                        names=['acc_x', 'acc_y', 'acc_z'],
                        dtype=np.float32,
                        sep=r'\s*,\s*',
                        engine='python'
                    )
                else:
                    sensor_name = file_path.stem.lower()
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=[sensor_name],
                        dtype=np.float32
                    )
                
                # Create datetime index
                timestamps = pd.date_range(
                    start=pd.to_datetime(initial_time, unit='s', utc=True),
                    periods=len(df),
                    freq=pd.DateOffset(seconds=1/sample_rate)
                )
                df.index = timestamps
                
                return (initial_time, sample_rate, df)

        except Exception as e:
            self.logger.error(f"Sensor load error: {str(e)}")
            return (0.0, 0.0, pd.DataFrame())

    def _process_data(self, subject_path: Path, exam_type: str, subject_id: int) -> pd.DataFrame:
        try:
            # Load sensor data
            sensors = {
                'bvp': self._load_sensor_data(subject_path/'BVP.csv'),
                'acc': self._load_sensor_data(subject_path/'ACC.csv'),
                'hr': self._load_sensor_data(subject_path/'HR.csv')
            }

            # Create DataFrames with column validation
            dfs = []
            for name, (time, rate, data) in sensors.items():
                df = self._create_timeseries(data, time, rate, name)
                if df.empty:
                    continue
                
                # Validate column count
                expected_cols = self._get_columns(name)
                if len(df.columns) != len(expected_cols):
                    self.logger.warning(f"Column mismatch in {name}. Expected {len(expected_cols)}, got {len(df.columns)}. Truncating.")
                    df = df.iloc[:, :len(expected_cols)]  # Take first N columns
                    df.columns = expected_cols
                
                dfs.append(df)

            # Merge with proper alignment
            merged = pd.concat(dfs, axis=1).assign(
                subject=subject_id,
                exam_type=exam_type,
                sampling_rate=self.target_rate
            )
            
            return merged.dropna(how='all').ffill()

        except Exception as e:
            self.logger.error(f"Error processing {subject_path}: {str(e)}")
            raise

    def _create_timeseries(self, data: pd.DataFrame, initial_time: float, 
                          sample_rate: float, sensor: str) -> pd.DataFrame:
        """Type-consistent time index creation"""
        if sensor == 'acc' and not data.empty:
            index = pd.date_range(
                start=pd.to_datetime(initial_time, unit='s', utc=True),
                periods=len(data),
                freq=pd.Timedelta(1/sample_rate, unit='s')
            )
            return data.set_index(index)
        # Existing handling for other sensors...

    def _resample_and_merge(self, dfs: dict, exam_type: str, subject_id: int) -> pd.DataFrame:
        """Resample and merge sensor data using base class methods"""
        valid_dfs = {
            name: (df, rate) 
            for name, (df, rate) in dfs.items() 
            if not df.empty
        }
        
        if not valid_dfs:
            self.logger.error(f"No valid data for {exam_type}")
            return pd.DataFrame()

        # Resample using base class functionality
        resampled = {
            name: self.resample_data(df, rate)
            for name, (df, rate) in valid_dfs.items()
        }

        # Merge with explicit column naming
        merged = pd.concat(
            [df.set_axis(self._get_columns(name), axis=1) 
             for name, df in resampled.items()],
            axis=1
        )

        # Add metadata
        merged['subject_id'] = subject_id
        merged['exam_type'] = exam_type
        merged['sampling_rate'] = self.target_rate
        
        return merged.dropna(how='all')

    def _get_columns(self, sensor: str) -> list:
        """Get expected columns for each sensor type"""
        return {
            'bvp': ['bvp'],
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'eda': ['eda'],
            'temp': ['temp'],
            'hr': ['hr']
        }[sensor]

    def _process_events(self, df: pd.DataFrame, subject_path: Path) -> pd.DataFrame:
        """Process event markers with error handling"""
        df['event'] = 0
        
        try:
            tags = pd.read_csv(subject_path/'tags.csv', header=None)
            tags.columns = ['timestamp']
            tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
            
            # Find nearest indices with tolerance
            event_indices = df.index.get_indexer(tags['timestamp'], method='nearest', tolerance='5s')
            valid_events = event_indices != -1
            
            if not np.all(valid_events):
                self.logger.warning(f"{sum(~valid_events)} events outside time range")
                
            df.iloc[event_indices[valid_events], df.columns.get_loc('event')] = 1
            
        except Exception as e:
            self.logger.error(f"Error processing events: {str(e)}")
            
        return df

    def get_labels(self) -> Dict[str, Any]:
        """Return complete label documentation"""
        return {
            'classes': {0: 'baseline', 1: 'stress'},
            'exam_durations': self.EXAM_DURATIONS,
            'description': (
                "Labels based on temporal positioning relative to exam period. "
                "1 = during exam, 0 = pre-exam baseline"
            ),
            'sensor_rates': self.SENSOR_RATES
        }

    def _validate_subject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate only sensor columns"""
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'temp', 'hr']
        
        # Validate ACC columns
        acc_data = df[sensor_cols[:3]].copy()
        acc_data /= 64  # Convert to g-forces
        
        # Per-axis validation (-2g to +2g range)
        axis_violations = (
            (acc_data < -2.0) | 
            (acc_data > 2.0)
        ).any(axis=1)
        
        # Calculate violation percentage
        violation_ratio = axis_violations.mean()
        
        if violation_ratio > 0.05:  # Allow up to 5% violations
            raise ValueError(f"Excessive ACC noise: {violation_ratio:.1%}")
        
        # Clean with explicit numeric handling
        acc_data.loc[axis_violations, sensor_cols[:3]] = np.nan
        
        # Update validated columns
        df[sensor_cols[:3]] = acc_data
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data before validation"""
        # Clean BVP
        df['bvp'] = self._clean_bvp(df['bvp'])
        
        # Clean HR
        if 'hr' in df.columns:
            df['hr'] = self._clean_hr(df['hr'])
        
        # Clean ACC
        df[self.ACC_COLS] = self._clean_acc(df[self.ACC_COLS])
        
        return df

    def _clean_bvp(self, bvp_series: pd.Series) -> pd.Series:
        """Advanced BVP cleaning pipeline"""
        # 1. Remove transient artifacts
        cleaned = bvp_series.rolling('5s').median()
        
        # 2. Normalize using adaptive scaling
        q99, q01 = cleaned.quantile(0.99), cleaned.quantile(0.01)
        normalized = (cleaned - q01) / (q99 - q01)  # Scale to 0-1 range
        
        # 3. Clip extreme values
        return normalized.clip(0, 1) * 2 - 1  # Scale to [-1, 1]

    def _clean_hr(self, hr_series: pd.Series) -> pd.Series:
        """Smooth HR using median filtering"""
        return hr_series.rolling('60s', min_periods=10).median()

    def _clean_acc(self, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme ACC values"""
        return acc_df.clip(-2.5, 2.5)

    def load_acc_only(self, subject_id: int, exam_type: str) -> pd.DataFrame:
        """Load only ACC data for analysis"""
        try:
            subj_path = self._validate_subject_path(subject_id, exam_type)
            acc_data = self._load_sensor_data(subj_path/'ACC.csv')
            return self._create_timeseries(acc_data, *self.SENSOR_RATES['acc'])
        except Exception as e:
            self.logger.error(f"ACC load failed: {str(e)}")
            return pd.DataFrame()

    def _validate_subject_path(self, subject_id: int, exam_type: str) -> Path:
        """Validate and return subject data path"""
        # Correct path structure based on actual dataset
        base_path = self.data_path
        subj_path = base_path / f"S{subject_id}" / exam_type
        
        if not subj_path.exists():
            raise FileNotFoundError(f"Subject {subject_id} data not found at {subj_path}")
        
        return subj_path