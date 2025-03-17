import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .base_feature_extractor import BaseFeatureExtractor
from .time_domain_extractor import TimeDomainExtractor
from .frequency_domain_extractor import FrequencyDomainExtractor
from .nonlinear_extractor import NonlinearExtractor
from .image_encoding_extractor import ImageEncodingExtractor
from .context_aware_extractor import ContextAwareExtractor
from .feature_selector import FeatureSelector

class FeatureEngineeringPipeline:
    """
    A pipeline for extracting features from physiological signals and selecting the most relevant ones.
    
    This pipeline combines multiple feature extractors and provides methods for:
    - Extracting features from raw signals
    - Selecting the most relevant features
    - Visualizing feature distributions and relationships
    - Saving features and metadata to disk
    """
    
    def __init__(self, window_size: int = 300, overlap: float = 0.5, 
                 sampling_rate: int = 30, output_dir: str = 'features',
                 device_info: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            window_size: Size of the window in samples (default: 300, which is 10s at 30Hz)
            overlap: Overlap between consecutive windows as a fraction (default: 0.5)
            sampling_rate: Sampling rate of the signal in Hz (default: 30)
            output_dir: Directory to save features and metadata (default: 'features')
            device_info: Dictionary containing device-specific information (default: None)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.output_dir = output_dir
        self.device_info = device_info or {}
        self.random_state = random_state
        
        # Initialize feature extractors
        self.time_extractor = TimeDomainExtractor(window_size, overlap, sampling_rate)
        self.freq_extractor = FrequencyDomainExtractor(window_size, overlap, sampling_rate)
        self.nonlinear_extractor = NonlinearExtractor(window_size, overlap, sampling_rate)
        self.image_extractor = ImageEncodingExtractor(window_size, overlap, sampling_rate)
        self.context_extractor = ContextAwareExtractor(window_size, overlap, sampling_rate, device_info)
        
        # Initialize feature selector
        self.feature_selector = FeatureSelector(random_state=random_state)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature metadata
        self.feature_metadata = {}
        
    def extract_features(self, df: pd.DataFrame, ppg_col: str,
                        acc_x_col: Optional[str] = None,
                        acc_y_col: Optional[str] = None,
                        acc_z_col: Optional[str] = None,
                        metadata_cols: Optional[List[str]] = None,
                        batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Extract features from a DataFrame containing physiological signals.
        
        Args:
            df: DataFrame containing the data
            ppg_col: Name of the column containing PPG data
            acc_x_col: Name of the column containing accelerometer X-axis data (optional)
            acc_y_col: Name of the column containing accelerometer Y-axis data (optional)
            acc_z_col: Name of the column containing accelerometer Z-axis data (optional)
            metadata_cols: List of column names containing metadata (optional)
            batch_size: Number of windows to process at once (optional, for memory efficiency)
            
        Returns:
            DataFrame containing extracted features
        """
        print(f"Extracting features from {len(df)} samples...")
        
        # Create windows from PPG data
        windows, indices = self.time_extractor.create_windows_from_df(df, ppg_col)
        
        print(f"Created {len(windows)} windows with {self.window_size} samples each and {self.overlap*100:.0f}% overlap")
        
        # Process in batches if specified
        if batch_size is not None:
            feature_dfs = []
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i+batch_size]
                batch_indices = indices[i:i+batch_size]
                batch_df = self._process_windows(df, batch_windows, batch_indices, ppg_col, 
                                               acc_x_col, acc_y_col, acc_z_col, metadata_cols)
                feature_dfs.append(batch_df)
                print(f"Processed batch {i//batch_size + 1}/{(len(windows)-1)//batch_size + 1}")
            
            # Combine batches
            features_df = pd.concat(feature_dfs, ignore_index=True)
        else:
            # Process all windows at once
            features_df = self._process_windows(df, windows, indices, ppg_col, 
                                              acc_x_col, acc_y_col, acc_z_col, metadata_cols)
        
        print(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} windows")
        
        return features_df
    
    def _process_windows(self, df: pd.DataFrame, windows: List[np.ndarray], 
                        indices: List[int], ppg_col: str,
                        acc_x_col: Optional[str] = None,
                        acc_y_col: Optional[str] = None,
                        acc_z_col: Optional[str] = None,
                        metadata_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process a batch of windows to extract features.
        
        Args:
            df: DataFrame containing the data
            windows: List of windows to process
            indices: List of starting indices for each window
            ppg_col: Name of the column containing PPG data
            acc_x_col: Name of the column containing accelerometer X-axis data (optional)
            acc_y_col: Name of the column containing accelerometer Y-axis data (optional)
            acc_z_col: Name of the column containing accelerometer Z-axis data (optional)
            metadata_cols: List of column names containing metadata (optional)
            
        Returns:
            DataFrame containing extracted features
        """
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
                end_idx = min(start_idx + self.window_size, len(df))
                if end_idx > start_idx:
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
            
            # Extract features from each extractor
            features = {}
            
            # Add window metadata
            features['window_idx'] = i
            features['start_idx'] = start_idx
            features['end_idx'] = min(start_idx + self.window_size, len(df))
            
            # Add timestamp if available
            if 'timestamp' in df.columns:
                features['timestamp'] = df['timestamp'].iloc[start_idx]
            
            # Add label if available
            if 'stress_level' in df.columns:
                features['stress_level'] = df['stress_level'].iloc[start_idx]
            
            # Extract time domain features
            time_features = self.time_extractor.extract_features(ppg_window)
            features.update({f'time_{k}': v for k, v in time_features.items()})
            
            # Extract frequency domain features
            freq_features = self.freq_extractor.extract_features(ppg_window)
            features.update({f'freq_{k}': v for k, v in freq_features.items()})
            
            # Extract nonlinear features
            nonlinear_features = self.nonlinear_extractor.extract_features(ppg_window)
            features.update({f'nonlinear_{k}': v for k, v in nonlinear_features.items()})
            
            # Extract image encoding features
            image_features = self.image_extractor.extract_features(ppg_window)
            features.update({f'image_{k}': v for k, v in image_features.items()})
            
            # Extract context-aware features if accelerometer data is available
            if acc_x_window is not None and acc_y_window is not None and acc_z_window is not None:
                context_features = self.context_extractor.extract_features(
                    ppg_window, acc_x_window, acc_y_window, acc_z_window, metadata
                )
                features.update({f'context_{k}': v for k, v in context_features.items()})
            
            feature_dicts.append(features)
        
        # Convert list of dictionaries to DataFrame
        features_df = pd.DataFrame(feature_dicts)
        
        return features_df
    
    def select_features(self, features_df: pd.DataFrame, target_col: str = 'stress_level',
                       n_features: int = 20) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Select the most relevant features for stress detection.
        
        Args:
            features_df: DataFrame containing extracted features
            target_col: Name of the column containing the target variable
            n_features: Number of features to select
            
        Returns:
            Tuple of (DataFrame with selected features, Dictionary of feature recommendations)
        """
        print(f"Selecting features from {features_df.shape[1]} total features...")
        
        # Separate features and target
        X = features_df.drop(columns=[target_col, 'window_idx', 'start_idx', 'end_idx', 'timestamp'], 
                           errors='ignore')
        y = features_df[target_col] if target_col in features_df.columns else None
        
        if y is None:
            print("Warning: Target column not found. Skipping feature selection.")
            return features_df, {}
        
        # Get feature recommendations
        recommendations = self.feature_selector.get_feature_recommendations(X, y, n_features=n_features)
        
        # Select consensus features
        selected_features = recommendations['consensus']
        
        # Add metadata columns back
        for col in ['window_idx', 'start_idx', 'end_idx', 'timestamp', target_col]:
            if col in features_df.columns:
                selected_features.append(col)
        
        # Create DataFrame with selected features
        selected_df = features_df[selected_features]
        
        print(f"Selected {len(selected_features)} features")
        
        return selected_df, recommendations
    
    def visualize_features(self, features_df: pd.DataFrame, target_col: str = 'stress_level',
                          output_dir: Optional[str] = None) -> None:
        """
        Visualize feature distributions and relationships.
        
        Args:
            features_df: DataFrame containing extracted features
            target_col: Name of the column containing the target variable
            output_dir: Directory to save visualizations (default: self.output_dir)
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"Generating visualizations in {vis_dir}...")
        
        # Separate features and target
        X = features_df.drop(columns=[target_col, 'window_idx', 'start_idx', 'end_idx', 'timestamp'], 
                           errors='ignore')
        y = features_df[target_col] if target_col in features_df.columns else None
        
        # Plot feature distributions
        self._plot_feature_distributions(X, y, vis_dir)
        
        # Plot correlation heatmap
        self._plot_correlation_heatmap(X, vis_dir)
        
        # Plot feature importance if target is available
        if y is not None:
            self._plot_feature_importance(X, y, vis_dir)
        
        print(f"Visualizations saved to {vis_dir}")
    
    def _plot_feature_distributions(self, X: pd.DataFrame, y: Optional[pd.Series],
                                  output_dir: str) -> None:
        """
        Plot distributions of top features.
        
        Args:
            X: DataFrame containing features
            y: Series containing target variable (optional)
            output_dir: Directory to save visualizations
        """
        # Select top features to visualize
        n_features = min(10, X.shape[1])
        
        if y is not None:
            # Use ANOVA F-value to select top features
            top_features = self.feature_selector.statistical_selection(X, y, method='f_classif', k=n_features)
        else:
            # Use variance to select top features
            variances = X.var().sort_values(ascending=False)
            top_features = variances.index[:n_features].tolist()
        
        # Create figure
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        
        # If only one feature, axes is not an array
        if n_features == 1:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(top_features):
            if y is not None:
                # Plot distribution by target class
                for class_val in y.unique():
                    sns.kdeplot(X[feature][y == class_val], label=f'Stress Level {class_val}', ax=axes[i])
                axes[i].legend()
            else:
                # Plot overall distribution
                sns.histplot(X[feature], kde=True, ax=axes[i])
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
        plt.close()
    
    def _plot_correlation_heatmap(self, X: pd.DataFrame, output_dir: str) -> None:
        """
        Plot correlation heatmap of features.
        
        Args:
            X: DataFrame containing features
            output_dir: Directory to save visualizations
        """
        # Select top features by variance
        n_features = min(20, X.shape[1])
        variances = X.var().sort_values(ascending=False)
        top_features = variances.index[:n_features].tolist()
        
        # Calculate correlation matrix
        corr = X[top_features].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
    
    def _plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, output_dir: str) -> None:
        """
        Plot feature importance.
        
        Args:
            X: DataFrame containing features
            y: Series containing target variable
            output_dir: Directory to save visualizations
        """
        # Select top features
        n_features = min(20, X.shape[1])
        
        # Get feature importances using Random Forest
        self.feature_selector.model_based_selection(X, y, n_features=n_features)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', 
                   data=self.feature_selector.feature_importances_.head(n_features))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    
    def save_features(self, features_df: pd.DataFrame, filename: str = 'features.csv',
                     metadata_filename: str = 'feature_metadata.json') -> None:
        """
        Save features and metadata to disk.
        
        Args:
            features_df: DataFrame containing features
            filename: Name of the CSV file to save features
            metadata_filename: Name of the JSON file to save metadata
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save features to CSV
        features_path = os.path.join(self.output_dir, filename)
        features_df.to_csv(features_path, index=False)
        
        print(f"Features saved to {features_path}")
        
        # Generate feature metadata
        metadata = self._generate_feature_metadata(features_df)
        
        # Save metadata to JSON
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Feature metadata saved to {metadata_path}")
    
    def _generate_feature_metadata(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for features.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Dictionary containing feature metadata
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': features_df.shape[0],
            'n_features': features_df.shape[1],
            'window_size': self.window_size,
            'overlap': self.overlap,
            'sampling_rate': self.sampling_rate,
            'features': {}
        }
        
        # Add metadata for each feature
        for column in features_df.columns:
            # Skip metadata columns
            if column in ['window_idx', 'start_idx', 'end_idx', 'timestamp', 'stress_level']:
                continue
            
            # Determine feature type
            if column.startswith('time_'):
                feature_type = 'time_domain'
                description = 'Time domain feature extracted from PPG signal'
            elif column.startswith('freq_'):
                feature_type = 'frequency_domain'
                description = 'Frequency domain feature extracted from PPG signal'
            elif column.startswith('nonlinear_'):
                feature_type = 'nonlinear'
                description = 'Nonlinear feature extracted from PPG signal'
            elif column.startswith('image_'):
                feature_type = 'image_encoding'
                description = 'Feature extracted from image encoding of PPG signal'
            elif column.startswith('context_'):
                feature_type = 'context_aware'
                description = 'Context-aware feature incorporating motion and metadata'
            else:
                feature_type = 'other'
                description = 'Other feature'
            
            # Calculate basic statistics
            stats = {
                'mean': float(features_df[column].mean()),
                'std': float(features_df[column].std()),
                'min': float(features_df[column].min()),
                'max': float(features_df[column].max()),
                'missing': int(features_df[column].isna().sum())
            }
            
            # Add feature metadata
            metadata['features'][column] = {
                'type': feature_type,
                'description': description,
                'statistics': stats
            }
        
        return metadata
    
    def run_pipeline(self, df: pd.DataFrame, ppg_col: str,
                    acc_x_col: Optional[str] = None,
                    acc_y_col: Optional[str] = None,
                    acc_z_col: Optional[str] = None,
                    metadata_cols: Optional[List[str]] = None,
                    target_col: str = 'stress_level',
                    batch_size: Optional[int] = None,
                    n_features: int = 20,
                    visualize: bool = True) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: DataFrame containing the data
            ppg_col: Name of the column containing PPG data
            acc_x_col: Name of the column containing accelerometer X-axis data (optional)
            acc_y_col: Name of the column containing accelerometer Y-axis data (optional)
            acc_z_col: Name of the column containing accelerometer Z-axis data (optional)
            metadata_cols: List of column names containing metadata (optional)
            target_col: Name of the column containing the target variable
            batch_size: Number of windows to process at once (optional, for memory efficiency)
            n_features: Number of features to select
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame containing selected features
        """
        print("Starting feature engineering pipeline...")
        
        # Extract features
        features_df = self.extract_features(df, ppg_col, acc_x_col, acc_y_col, acc_z_col, 
                                          metadata_cols, batch_size)
        
        # Save all features
        self.save_features(features_df, filename='all_features.csv')
        
        # Select features if target column is available
        if target_col in df.columns:
            selected_df, recommendations = self.select_features(features_df, target_col, n_features)
            
            # Save selected features
            self.save_features(selected_df, filename='selected_features.csv')
            
            # Save feature recommendations
            recommendations_path = os.path.join(self.output_dir, 'feature_recommendations.json')
            with open(recommendations_path, 'w') as f:
                json.dump({k: v for k, v in recommendations.items()}, f, indent=2)
            
            print(f"Feature recommendations saved to {recommendations_path}")
        else:
            selected_df = features_df
            print("Warning: Target column not found. Skipping feature selection.")
        
        # Generate visualizations
        if visualize:
            self.visualize_features(features_df, target_col)
        
        print("Feature engineering pipeline completed successfully!")
        
        return selected_df 