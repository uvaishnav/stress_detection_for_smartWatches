import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the feature_engineering module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_pipeline import FeatureEngineeringPipeline

def load_sample_data(file_path: str) -> pd.DataFrame:
    """
    Load sample data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def main():
    """
    Main function to demonstrate the feature engineering pipeline.
    """
    # Set paths
    data_path = "../../data/processed/cleaned_ppg_data.csv"  # Adjust path as needed
    output_dir = "../../features"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = load_sample_data(data_path)
    except FileNotFoundError:
        print(f"Error: File {data_path} not found.")
        print("Generating synthetic data for demonstration...")
        
        # Generate synthetic data for demonstration
        n_samples = 10000
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='100ms'),
            'ppg': np.sin(np.linspace(0, 100*np.pi, n_samples)) + 0.1 * np.random.randn(n_samples),
            'acc_x': 0.5 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + 0.05 * np.random.randn(n_samples),
            'acc_y': 0.5 * np.cos(np.linspace(0, 10*np.pi, n_samples)) + 0.05 * np.random.randn(n_samples),
            'acc_z': 0.5 * np.sin(np.linspace(0, 5*np.pi, n_samples)) + 0.05 * np.random.randn(n_samples),
            'device_type': 'apple_watch',
            'stress_level': np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
        })
        
        # Save synthetic data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Saved synthetic data to {data_path}")
    
    # Print data info
    print("\nData preview:")
    print(df.head())
    
    print("\nData columns:")
    print(df.columns.tolist())
    
    print("\nData statistics:")
    print(df.describe())
    
    # Define device info
    device_info = {
        'device_type': 'apple_watch',
        'sensor_quality': 0.9,
        'wearing_position': 'wrist_top'
    }
    
    # Initialize feature engineering pipeline
    pipeline = FeatureEngineeringPipeline(
        window_size=300,  # 10 seconds at 30 Hz
        overlap=0.5,      # 50% overlap
        sampling_rate=30, # 30 Hz
        output_dir=output_dir,
        device_info=device_info,
        random_state=42
    )
    
    # Define column names
    ppg_col = 'ppg'
    acc_x_col = 'acc_x' if 'acc_x' in df.columns else None
    acc_y_col = 'acc_y' if 'acc_y' in df.columns else None
    acc_z_col = 'acc_z' if 'acc_z' in df.columns else None
    metadata_cols = ['device_type'] if 'device_type' in df.columns else None
    target_col = 'stress_level' if 'stress_level' in df.columns else None
    
    # Run the pipeline
    selected_features = pipeline.run_pipeline(
        df=df,
        ppg_col=ppg_col,
        acc_x_col=acc_x_col,
        acc_y_col=acc_y_col,
        acc_z_col=acc_z_col,
        metadata_cols=metadata_cols,
        target_col=target_col,
        batch_size=100,  # Process 100 windows at a time
        n_features=20,   # Select top 20 features
        visualize=True   # Generate visualizations
    )
    
    # Print selected features
    print("\nSelected features:")
    print(selected_features.columns.tolist())
    
    print("\nFeature engineering completed successfully!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 