# Feature Engineering Module

This module provides a comprehensive framework for extracting features from physiological signals (PPG) and accelerometer data for stress detection. It implements various feature extraction techniques across multiple domains and provides tools for feature selection and visualization.

## Overview

The feature engineering module consists of the following components:

1. **Base Feature Extractor**: Provides common functionality for windowing and feature extraction.
2. **Domain-Specific Extractors**:
   - **Time Domain Extractor**: Extracts statistical and HRV features from the time domain.
   - **Frequency Domain Extractor**: Extracts spectral features from the frequency domain.
   - **Nonlinear Extractor**: Extracts entropy and complexity measures.
   - **Image Encoding Extractor**: Implements GASF and MTF encodings for image-based features.
   - **Context-Aware Extractor**: Incorporates motion data and device-specific information.
3. **Feature Selector**: Provides methods for selecting the most relevant features.
4. **Feature Pipeline**: Orchestrates the entire feature engineering process.

## Installation

The module requires the following dependencies:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

## Usage

### Basic Usage

```python
from feature_engineering.feature_pipeline import FeatureEngineeringPipeline

# Initialize the pipeline
pipeline = FeatureEngineeringPipeline(
    window_size=300,  # 10 seconds at 30 Hz
    overlap=0.5,      # 50% overlap
    sampling_rate=30, # 30 Hz
    output_dir='features',
    device_info={'device_type': 'apple_watch'}
)

# Run the pipeline
selected_features = pipeline.run_pipeline(
    df=data,
    ppg_col='ppg',
    acc_x_col='acc_x',
    acc_y_col='acc_y',
    acc_z_col='acc_z',
    metadata_cols=['device_type'],
    target_col='stress_level',
    batch_size=100,
    n_features=20,
    visualize=True
)
```

### Advanced Usage

You can also use the individual components separately:

```python
# Extract time domain features
from feature_engineering.time_domain_extractor import TimeDomainExtractor

extractor = TimeDomainExtractor(window_size=300, overlap=0.5, sampling_rate=30)
windows, indices = extractor.create_windows_from_df(df, 'ppg')
features = extractor.extract_features_from_windows(windows)
```

## Feature Categories

### Time Domain Features
- Statistical features (mean, std, min, max, etc.)
- Peak-based features (peak count, mean peak height, etc.)
- Heart Rate Variability (HRV) features (SDNN, RMSSD, pNN50, etc.)

### Frequency Domain Features
- Band powers (VLF, LF, HF)
- Spectral characteristics (centroid, spread, skewness, kurtosis)
- Dominant frequencies and their powers

### Nonlinear Features
- Entropy measures (Sample Entropy, Approximate Entropy, Permutation Entropy)
- Poincaré plot features (SD1, SD2, SD1/SD2 ratio)
- Detrended Fluctuation Analysis (DFA) features
- Recurrence Quantification Analysis (RQA) features

### Image Encoding Features
- Gramian Angular Summation Field (GASF) encoding
- Markov Transition Field (MTF) encoding
- Statistical features derived from these encodings

### Context-Aware Features
- Motion-related features from accelerometer data
- Correlation between PPG and motion
- Device-specific normalization
- Metadata-based features (time of day, user demographics, etc.)

## Feature Selection

The module provides several methods for feature selection:
- Correlation-based selection
- Statistical tests (ANOVA F-value, mutual information)
- Model-based selection (Random Forest importance)
- Dimensionality reduction (PCA)

## Output

The pipeline generates the following outputs:
- CSV files with extracted features
- JSON files with feature metadata
- Visualizations of feature distributions and relationships
- Feature recommendations for different model architectures

## Example

See `example.py` for a complete example of how to use the feature engineering pipeline.

## References

- Gramian Angular Field (GAF): Z. Wang and T. Oates, "Imaging Time-Series to Improve Classification and Imputation," in IJCAI, 2015.
- Markov Transition Field (MTF): Z. Wang and T. Oates, "Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks," in AAAI Workshop, 2015.
- Heart Rate Variability (HRV): Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, "Heart rate variability: standards of measurement, physiological interpretation and clinical use," Circulation, vol. 93, no. 5, pp. 1043-1065, 1996. 