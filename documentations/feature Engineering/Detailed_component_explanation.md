
# 📚 5. Detailed Component Documentation

## 🛠️ 5.1 BaseFeatureExtractor

- **Purpose:**  
  The `BaseFeatureExtractor` is an abstract base class that defines the common interface and functionality for all feature extractors in the system. It provides standard methods for creating sliding windows from time series data and extracting features from those windows, ensuring consistent behavior across different feature extraction techniques.

- **Usage Example:**  
  ```python
  # BaseFeatureExtractor cannot be instantiated directly as it's an abstract class
  # Instead, a concrete subclass would be used like this:
  extractor = TimeDomainExtractor(window_size=300, overlap=0.5, sampling_rate=30)
  windows = extractor.create_windows(ppg_signal)
  features_df = extractor.extract_features_from_windows(windows)
  ```

- **Parameters & Returns:**  
  | Parameter    | Type            | Description                                                  |
  |--------------|-----------------|--------------------------------------------------------------|
  | window_size  | int             | Size of the window in samples (default: 300, which is 10s at 30Hz) |
  | overlap      | float           | Overlap between consecutive windows as a fraction (default: 0.5) |
  | **Returns**  |                 |                                                              |
  | Windows      | List[np.ndarray]| List of numpy arrays, each representing a window of the signal |
  | Features     | pd.DataFrame    | DataFrame containing the extracted features, with one row per window |

- **Internal Workflow:**  
  - 📋 Calculates the appropriate step size based on window size and overlap  
  - 🌐 Creates overlapping windows from the input data  
  - 🛠️ Provides a framework for feature extraction from these windows  
  - 🔍 The abstract `extract_features` method must be implemented by subclasses to define domain-specific feature extraction logic  

- **Dependencies:**  
  - `numpy` for array operations  
  - `pandas` for DataFrame manipulation  
  - `ABC` and `abstractmethod` from the `abc` module for defining the abstract interface  
  - `typing` module for type hints  

---

## ⏳ 5.2 TimeDomainExtractor

- **Purpose:**  
  Extracts time-domain features from physiological signals, including statistical measures (mean, standard deviation, etc.) and heart rate variability (HRV) features that capture temporal patterns in the data.

- **Usage Example:**  
  ```python
  # Initialize extractor with appropriate parameters
  extractor = TimeDomainExtractor(window_size=300, overlap=0.5, sampling_rate=30)
  
  # Extract features from a single window
  features = extractor.extract_features(ppg_window)
  
  # Or process an entire DataFrame
  features_df = extractor.process_dataframe(df, columns=['ppg_signal'])
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                              |
  |---------------|--------------|----------------------------------------------------------|
  | window_size   | int          | Size of the window in samples (default: 300)             |
  | overlap       | float        | Overlap between consecutive windows (default: 0.5)       |
  | sampling_rate | int          | Sampling rate of the signal in Hz (default: 30)          |
  | **Returns**   |              |                                                          |
  | features      | Dict[str, float] | Dictionary of feature names and values                |
  | features_df   | pd.DataFrame | DataFrame with all extracted features, one row per window |

- **Internal Workflow:**  
  1. 🌟 Divides the input signal into overlapping windows  
  2. ⚙️ For each window, it calculates:  
     - Basic statistical features (mean, std, median, etc.)  
     - Features based on peaks in the signal  
     - Heart rate variability features if sufficient peaks are detected  
  3. 📊 Statistical features quantify the central tendency and dispersion of the signal  
  4. ⛰️ Peak-based features identify and analyze peaks in the signal, which often correspond to heartbeats  
  5. ❤️ HRV features such as SDNN, RMSSD, and pNN50 capture variations in the intervals between consecutive heartbeats  

- **Dependencies:**  
  - Inherits from `BaseFeatureExtractor`  
  - `numpy` for numerical operations  
  - `pandas` for data manipulation  
  - `scipy.stats` for statistical calculations  
  - `scipy.signal.find_peaks` for peak detection  

---

## 🌊 5.3 FrequencyDomainExtractor

- **Purpose:**  
  Extracts frequency-domain features from physiological signals, focusing on spectral power in different frequency bands, spectral ratios, and other spectral characteristics that are highly relevant for stress detection.

- **Usage Example:**  
  ```python
  # Initialize extractor
  extractor = FrequencyDomainExtractor(window_size=300, overlap=0.5, sampling_rate=30)
  
  # Extract features from a single window
  features = extractor.extract_features(ppg_window)
  
  # Process windows from a DataFrame
  features_df = extractor.extract_features_from_windows(windows)
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                      |
  |---------------|--------------|--------------------------------------------------|
  | window_size   | int          | Size of the window in samples                    |
  | overlap       | float        | Overlap between consecutive windows              |
  | sampling_rate | int          | Sampling rate of the signal in Hz                |
  | **Returns**   |              |                                                  |
  | features      | Dict[str, float] | Dictionary of frequency-domain features       |

- **Internal Workflow:**  
  1. 🧹 Detrends the signal to remove low-frequency trends  
  2. 📈 Calculates the power spectral density (PSD) using Welch's method  
  3. 🎨 Extracts power in specific frequency bands defined by physiological standards:  
     - Very Low Frequency (VLF): 0.0033-0.04 Hz  
     - Low Frequency (LF): 0.04-0.15 Hz  
     - High Frequency (HF): 0.15-0.4 Hz  
     - Cardiac band: 0.8-4.0 Hz  
  4. 📊 Calculates spectral characteristics such as spectral centroid, spread, skewness, and kurtosis  
  5. 🔝 Identifies dominant frequencies and their powers  
  6. ⚖️ Computes spectral entropy to measure randomness in the frequency distribution  

- **Dependencies:**  
  - Inherits from `BaseFeatureExtractor`  
  - `numpy` for array operations  
  - `pandas` for DataFrame handling  
  - `scipy.signal` for signal processing functions  
  - `scipy.fft` for Fast Fourier Transform operations  

---

## 🔮 5.4 NonlinearExtractor

- **Purpose:**  
  Extracts non-linear features from physiological signals, including entropy measures, Poincaré plot features, and complexity metrics that capture the irregularity and complexity patterns associated with stress responses.

- **Usage Example:**  
  ```python
  # Initialize the extractor
  extractor = NonlinearExtractor(window_size=300, overlap=0.5, sampling_rate=30)
  
  # Extract features from a single window
  features = extractor.extract_features(ppg_window)
  
  # Process a DataFrame
  processed_df = extractor.process_dataframe(df, columns=['ppg_signal'])
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                      |
  |---------------|--------------|--------------------------------------------------|
  | window_size   | int          | Size of the window in samples                    |
  | overlap       | float        | Overlap between consecutive windows              |
  | sampling_rate | int          | Sampling rate of the signal in Hz                |
  | **Returns**   |              |                                                  |
  | features      | Dict[str, float] | Dictionary of non-linear features             |

- **Internal Workflow:**  
  1. 🌌 Calculates entropy-based features including:  
     - Sample Entropy to quantify signal predictability  
     - Approximate Entropy for measuring regularity  
     - Permutation Entropy to capture ordinal patterns  
  2. 📊 Extracts Poincaré plot features by first finding peaks in the signal:  
     - SD1 (standard deviation perpendicular to line of identity)  
     - SD2 (standard deviation along line of identity)  
     - SD1/SD2 ratio and elliptical area  
     - Cardiac Sympathetic Index (CSI) and Cardiac Vagal Index (CVI)  
  3. 🔄 Computes Detrended Fluctuation Analysis (DFA) features:  
     - Alpha1 for short-term correlations  
     - Alpha2 for long-term correlations  
  4. 🔍 Calculates Recurrence Quantification Analysis (RQA) features:  
     - Recurrence rate and determinism  

- **Dependencies:**  
  - Inherits from `BaseFeatureExtractor`  
  - `numpy` for numerical operations  
  - `pandas` for data structures  
  - `scipy.signal.find_peaks` for peak detection  

---

## 🌍 5.5 ContextAwareExtractor

- **Purpose:**  
  Extracts context-aware features that incorporate motion data and device-specific information to improve the accuracy of stress detection. This extractor focuses on handling motion artifacts and device variations that can affect physiological signals.

- **Usage Example:**  
  ```python
  # Initialize with device-specific information
  device_info = {'device_type': 'apple_watch', 'sensor_quality': 0.95}
  extractor = ContextAwareExtractor(window_size=300, overlap=0.5, 
                                    sampling_rate=30, device_info=device_info)
  
  # Extract features with accelerometer data
  features = extractor.extract_features(ppg_window, 
                                      acc_x=acc_x_window, 
                                      acc_y=acc_y_window, 
                                      acc_z=acc_z_window)
  
  # Process a DataFrame with context information
  features_df = extractor.process_dataframe_with_context(
      df, ppg_col='ppg', acc_x_col='acc_x', acc_y_col='acc_y', acc_z_col='acc_z',
      metadata_cols=['time_of_day', 'temperature'])
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                      |
  |---------------|--------------|--------------------------------------------------|
  | window_size   | int          | Size of the window in samples                    |
  | overlap       | float        | Overlap between consecutive windows              |
  | sampling_rate | int          | Sampling rate of the signal in Hz                |
  | device_info   | Dict[str, Any] | Dictionary containing device-specific information |
  | **Returns**   |              |                                                  |
  | features      | Dict[str, float] | Dictionary of context-aware features          |

- **Internal Workflow:**  
  1. 🏃 Extracts motion-related features from accelerometer data:  
     - Statistical features for each axis (mean, std, range, etc.)  
     - Activity level estimation from acceleration magnitude  
     - Jerk (derivative of acceleration) features  
     - Dominant frequency analysis  
  2. 🔗 Calculates correlation between PPG and motion:  
     - Correlation coefficients between PPG and each acceleration axis  
     - Motion artifact probability estimation  
     - Signal quality estimation  
     - Cross-correlation at different lags  
  3. 🛠️ Incorporates device-specific normalization:  
     - Applies device-type specific adjustment factors  
     - Accounts for sensor quality and wearing position  
  4. ⏰ Integrates metadata features:  
     - Time of day factors (circadian rhythm effects)  
     - User demographic factors (age, gender)  
     - Environmental factors (temperature)  

- **Dependencies:**  
  - Inherits from `BaseFeatureExtractor`  
  - `numpy` for array operations  
  - `pandas` for data manipulation  
  - `scipy.stats` for statistical calculations  
  - Custom peak finding method  

---

## 🖼️ 5.6 ImageEncodingExtractor

- **Purpose:**  
  Transforms time series data into image-based representations that capture temporal patterns in a visual format, enabling the use of computer vision techniques for stress detection. Implements both Gramian Angular Summation Field (GASF) and Markov Transition Field (MTF) encodings.

- **Usage Example:**  
  ```python
  # Initialize the extractor
  extractor = ImageEncodingExtractor(window_size=300, overlap=0.5, 
                                     sampling_rate=30, image_size=24)
  
  # Extract features from a window
  features = extractor.extract_features(ppg_window)
  
  # Generate image encodings for visualization or CNN input
  gasf, mtf = extractor.generate_image_encodings(ppg_window)
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                      |
  |---------------|--------------|--------------------------------------------------|
  | window_size   | int          | Size of the window in samples                    |
  | overlap       | float        | Overlap between consecutive windows              |
  | sampling_rate | int          | Sampling rate of the signal in Hz                |
  | image_size    | int          | Size of the resulting image encoding             |
  | **Returns**   |              |                                                  |
  | features      | Dict[str, float] | Dictionary of features derived from image encodings |
  | (gasf, mtf)   | Tuple[np.ndarray, np.ndarray] | Image encodings as 2D arrays      |

- **Internal Workflow:**  
  1. 📏 Resamples the input window to the desired image size  
  2. 🎨 Generates GASF encoding:  
     - Normalizes the signal to [-1, 1]  
     - Converts to polar coordinates (angles)  
     - Computes the GASF matrix using cosine of sum of angles  
  3. 🔢 Generates MTF encoding:  
     - Quantizes the signal into discrete bins  
     - Calculates Markov transition probabilities between quantized states  
     - Creates the MTF matrix representing these transitions  
  4. 📊 Extracts statistical features from both encodings:  
     - Basic statistics (mean, std, min, max)  
     - Percentile-based features  
     - Entropy and energy measures  
     - Diagonal and asymmetry features  

- **Dependencies:**  
  - Inherits from `BaseFeatureExtractor`  
  - `numpy` for numerical operations  
  - `pandas` for data structures  
  - `scipy.signal` for signal processing  

---

## 🌟 5.7 FeatureSelector

- **Purpose:**  
  Provides methods for selecting the most relevant features from a feature matrix using various techniques, including correlation analysis, statistical tests, model-based selection, and dimensionality reduction.

- **Usage Example:**  
  ```python
  # Initialize selector
  selector = FeatureSelector(random_state=42)
  
  # Use correlation-based selection
  selected_features = selector.correlation_selection(features_df, threshold=0.8)
  
  # Use model-based selection with Random Forest
  important_features = selector.model_based_selection(X, y, n_features=20)
  
  # Get comprehensive feature recommendations
  recommendations = selector.get_feature_recommendations(X, y, n_features=20)
  ```

- **Parameters & Returns:**  
  | Parameter       | Type         | Description                                      |
  |-----------------|--------------|--------------------------------------------------|
  | random_state    | int          | Random seed for reproducibility                  |
  | threshold       | float        | Correlation threshold for feature selection      |
  | n_features      | int          | Number of features to select                     |
  | **Returns**     |              |                                                  |
  | selected_features | List[str]  | List of selected feature names                  |
  | recommendations | Dict[str, List[str]] | Dictionary mapping selection methods to recommended features |
  | pca_df          | pd.DataFrame | DataFrame containing PCA-transformed features   |

- **Internal Workflow:**  
  1. 📉 Correlation-based selection:  
     - Calculates correlation matrix between all features  
     - Removes features with correlation above threshold to reduce redundancy  
  2. 📊 Statistical selection:  
     - Applies statistical tests (ANOVA F-value or mutual information)  
     - Ranks features by their statistical significance with the target variable  
  3. 🌳 Model-based selection:  
     - Uses Random Forest feature importance or Recursive Feature Elimination  
     - Ranks features based on their contribution to model performance  
  4. 🔍 PCA selection:  
     - Applies Principal Component Analysis for dimensionality reduction  
     - Transforms features to principal components that explain variance  
  5. ✅ Feature recommendations:  
     - Combines results from multiple selection methods  
     - Creates a consensus list of features appearing in multiple methods  

- **Dependencies:**  
  - `numpy` for numerical operations  
  - `pandas` for data manipulation  
  - `sklearn.feature_selection` for selection methods  
  - `sklearn.ensemble.RandomForestClassifier` for importance-based selection  
  - `sklearn.decomposition.PCA` for dimensionality reduction  
  - `matplotlib` and `seaborn` for visualization  

---

## ⚙️ 5.8 FeaturePipeline

- **Purpose:**  
  Orchestrates the entire feature engineering process, coordinating multiple feature extractors, managing batch processing of large datasets, performing feature selection, and visualizing the results.

- **Usage Example:**  
  ```python
  # Initialize the pipeline with appropriate parameters
  pipeline = FeatureEngineeringPipeline(
      window_size=300, overlap=0.5, sampling_rate=30,
      output_dir='features', device_info={'device_type': 'fitbit'}
  )
  
  # Run the full pipeline
  features_df = pipeline.run_pipeline(
      df, ppg_col='ppg_signal',
      acc_x_col='acc_x', acc_y_col='acc_y', acc_z_col='acc_z',
      target_col='stress_level', n_features=20, visualize=True
  )
  
  # Or extract features only
  features_df = pipeline.extract_features(
      df, ppg_col='ppg_signal',
      acc_x_col='acc_x', acc_y_col='acc_y', acc_z_col='acc_z'
  )
  ```

- **Parameters & Returns:**  
  | Parameter     | Type         | Description                                      |
  |---------------|--------------|--------------------------------------------------|
  | window_size   | int          | Size of the window in samples                    |
  | overlap       | float        | Overlap between consecutive windows              |
  | sampling_rate | int          | Sampling rate of the signal in Hz                |
  | output_dir    | str          | Directory to save feature files and visualizations |
  | device_info   | Dict[str, Any] | Device-specific information                   |
  | **Returns**   |              |                                                  |
  | features_df   | pd.DataFrame | DataFrame containing all extracted features      |
  | selected_features | pd.DataFrame | DataFrame with only the selected features    |

- **Internal Workflow:**  
  1. 🚀 Initializes all feature extractors with consistent parameters  
  2. 🌟 Extracts features from the input data:  
     - Time-domain features from TimeDomainExtractor  
     - Frequency-domain features from FrequencyDomainExtractor  
     - Non-linear features from NonlinearExtractor  
     - Context-aware features from ContextAwareExtractor  
     - Image encoding features from ImageEncodingExtractor  
  3. 📦 Processes data in batches for memory efficiency if needed  
  4. 🔗 Combines features from all extractors into a single DataFrame  
  5. 🌟 Performs feature selection to identify the most relevant features  
  6. 📊 Generates visualizations for feature analysis:  
     - Feature distributions  
     - Correlation heatmaps  
     - Feature importance plots  
  7. 💾 Saves the extracted features and metadata to disk  

- **Dependencies:**  
  - All feature extractor classes  
  - `FeatureSelector` for feature selection  
  - `numpy` and `pandas` for data manipulation  
  - `sklearn` for machine learning utilities  
  - `matplotlib` and `seaborn` for visualization  
  - Standard Python libraries for file operations  
