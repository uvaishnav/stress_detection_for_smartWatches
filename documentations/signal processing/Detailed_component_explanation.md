## 📄 Detailed Component Documentation

### ⚙️ BaseSignalProcessor
- **Purpose:**  
  Serves as the foundation for signal processing by loading and providing access to the unified dataset. This component handles the initial data ingestion from Parquet files, which contain time-series physiological data from wearable devices.

- **Usage Example:**  
  ```python
  # Initialize the processor with path to unified dataset
  processor = BaseSignalProcessor("data/unified_dataset.parquet")
  
  # Load the dataset into memory
  dataset = processor.load_data()
  
  # Now dataset can be passed to other processing components
  print(f"Loaded dataset with {len(dataset)} samples")
  ```

- **Parameters & Returns:**

  | Parameter | Type | Description |
  |-----------|------|-------------|
  | data_path | str | Path to the unified Parquet file containing physiological and motion data |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | dataset | pd.DataFrame | Loaded dataset with columns for PPG signals, accelerometer data, and metadata |

> **Note:** The component initializes with a path to the Parquet file. When `load_data()` is called, it reads the Parquet file into a pandas DataFrame, which provides efficient columnar storage for time-series data. The dataset is stored as an instance variable for subsequent access by processing components. The loading process is designed to be memory-efficient, leveraging Parquet's columnar format to load only necessary columns when needed.

- **Dependencies:**  
  - `pandas`: For DataFrame handling and Parquet file reading

### 🔄 AdaptiveFilter
- **Purpose:**  
  Implements an adaptive Least Mean Squares (LMS) filter with spectral subtraction to remove motion artifacts from PPG signals. This component uses accelerometer data as a reference signal to identify and eliminate motion-induced noise while preserving cardiac components.

- **Usage Example:**  
  ```python
  # Initialize the adaptive filter with custom parameters
  adaptive_filter = AdaptiveFilter(learning_rate=0.001, filter_length=30)
  
  # Apply filtering to remove motion artifacts
  cleaned_signal = adaptive_filter.apply_adaptive_filter(
      noisy_signal=ppg_data,
      reference_signal=accelerometer_magnitude,
      motion_burst=motion_flags
  )
  
  # The cleaned_signal now has reduced motion artifacts
  ```

- **Parameters & Returns:**

  | Parameter | Type | Description |
  |-----------|------|-------------|
  | learning_rate | float | Controls adaptation speed of filter coefficients (default: 0.001) |
  | filter_length | int | Number of filter taps/coefficients (default: 30) |
  
  | Method Parameter | Type | Description |
  |------------------|------|-------------|
  | noisy_signal | np.ndarray | PPG signal with motion artifacts |
  | reference_signal | np.ndarray | Accelerometer magnitude as noise reference |
  | motion_burst | np.ndarray | Binary array indicating motion periods (0 or 1) |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | filtered_signal | np.ndarray | Cleaned PPG signal with preserved cardiac components |

> **Workflow Overview:**  
> 1. The filter first applies bandpass filtering (0.8-4 Hz) to the reference signal to match cardiac frequency range.
> 2. It performs frequency-domain processing with FFT to enable spectral subtraction.
> 3. Motion-adaptive spectral subtraction is applied with extremely conservative ratios to protect cardiac information.
> 4. The filter then implements time-domain adaptive filtering using the normalized LMS algorithm:
>    - For each sample, it calculates the filter output using current coefficients.
>    - Computes the error between the desired and actual output.
>    - Updates coefficients based on the error, reference signal, and learning rate.
>    - Applies gradient clipping for stability.
> 5. Physiological constraints are enforced to ensure the output remains within plausible biological limits.
> 6. Cardiac components are extracted and enhanced to preserve pulse information.
> 7. The final signal blends the filtered output with the original signal and enhanced cardiac component.

- **Dependencies:**  
  - `numpy`: For array operations and mathematical functions
  - `scipy.signal`: For bandpass filtering, peak finding, and Hilbert transform
  - `logging`: For diagnostic information

### 📉 KalmanFilter
- **Purpose:**  
  Provides temporal smoothing of physiological signals using a motion-aware Kalman filter. This component adaptively adjusts filter parameters based on motion detection to optimally balance noise reduction and signal preservation during both stable and motion periods.

- **Usage Example:**  
  ```python
  # Initialize the Kalman filter with custom parameters
  kalman_filter = KalmanFilter(
      initial_state=0.0,
      process_noise=1e-2,
      measurement_noise=1e-2
  )
  
  # Apply to an entire signal with motion burst information
  smoothed_signal = kalman_filter.apply_kalman_filter(
      signal=ppg_cleaned,
      motion_burst=motion_flags
  )
  
  # Or update sample by sample
  filtered_value = kalman_filter.update(
      measurement=current_sample,
      motion_burst=is_motion_present
  )
  ```

- **Parameters & Returns:**

  | Parameter | Type | Description |
  |-----------|------|-------------|
  | initial_state | float | Initial state estimate (default: 0.0) |
  | process_noise | float | Process noise covariance (Q) (default: 1e-2) |
  | measurement_noise | float | Measurement noise covariance (R) (default: 1e-2) |
  
  | Method Parameter | Type | Description |
  |------------------|------|-------------|
  | measurement | float | Current PPG sample value |
  | motion_burst | bool | Flag indicating if sample is during motion |
  | signal | np.ndarray | Complete signal to filter |
  | motion_burst | np.ndarray | Binary array of motion indicators |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | state | float | Filtered PPG value (from update method) |
  | filtered | np.ndarray | Complete filtered signal (from apply_kalman_filter) |

> **Workflow Overview:**  
> 1. The filter maintains internal state (current estimate) and error covariance.
> 2. During updates, it dynamically adjusts process noise based on motion detection:
>    - Higher process noise during motion (2e-1) allows faster adaptation.
>    - Lower process noise during stable periods (1e-2) provides smoother output.
> 3. The prediction step projects the current state and error covariance forward.
> 4. The update step:
>    - Calculates innovation (difference between measurement and prediction).
>    - Computes Kalman gain based on error covariance and measurement noise.
>    - Applies physiological plausibility checks to limit implausible innovations.
>    - Updates state estimate using weighted innovation.
>    - Updates error covariance.
> 5. The filter maintains history of innovations and states for adaptive processing.
> 6. When processing complete signals, it:
>    - Initializes with the first signal value.
>    - Processes each sample sequentially.
>    - Applies minimal smoothing for stability.
>    - Blends with original signal to preserve characteristics.
>    - Rescales to match original amplitude characteristics.

- **Dependencies:**  
  - `numpy`: For array operations and mathematical functions

### 🚨 MotionArtifactDetector
- **Purpose:**  
  Detects periods of motion in physiological signals using accelerometer data. This component implements device-aware normalization and a state machine approach to identify motion bursts that may contaminate PPG signals, enabling targeted artifact removal by downstream components.

- **Usage Example:**  
  ```python
  # Initialize the detector with custom parameters
  detector = MotionArtifactDetector(
      acc_threshold_factor=1.3,
      burst_duration=1.5,
      sampling_rate=30
  )
  
  # Detect motion bursts in the dataset
  dataset_with_motion = detector.detect_motion_bursts(dataset)
  
  # Access the motion burst flags
  motion_flags = dataset_with_motion['motion_burst']
  ```

- **Parameters & Returns:**

  | Parameter | Type | Description |
  |-----------|------|-------------|
  | acc_threshold_factor | float | Factor to scale the median accelerometer magnitude (default: 1.3) |
  | burst_duration | float | Duration of motion bursts in seconds (default: 1.5) |
  | sampling_rate | int | Sampling rate of the dataset in Hz (default: 30) |
  
  | Method Parameter | Type | Description |
  |------------------|------|-------------|
  | dataset | pd.DataFrame | Dataset containing accelerometer and device data |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | dataset | pd.DataFrame | Original dataset with added 'motion_burst' column (binary) |

> **Workflow Overview:**  
> 1. The detector first applies device-specific normalization to accelerometer data:
>    - Maps device names to appropriate scaling factors (e.g., Apple Watch: 2048, Galaxy Watch: 1024).
>    - Normalizes accelerometer values based on device-specific scales.
> 2. Calculates accelerometer magnitude using the Euclidean norm of x, y, z components.
> 3. Normalizes magnitude using robust statistics (median and IQR).
> 4. Computes a dynamic threshold based on:
>    - Median accelerometer magnitude.
>    - Standard deviation scaled by noise level.
> 5. Implements a state machine with persistence to detect motion bursts:
>    - Uses sigmoid-based attack and decay rates for smooth transitions.
>    - Increases state value when acceleration exceeds threshold.
>    - Decreases state value when acceleration is below threshold.
>    - Attack and decay rates adapt based on current state.
> 6. Applies final quantization with hysteresis to produce binary motion burst flags.
> 7. Adds the motion_burst column to the dataset.

- **Dependencies:**  
  - `numpy`: For array operations and mathematical functions
  - `pandas`: For DataFrame handling
  - `logging`: For diagnostic information

### 🌊 WaveletDenoiser
- **Purpose:**  
  Performs advanced signal denoising using wavelet transforms with skin-tone adaptive parameters. This component decomposes signals into frequency subbands, applies targeted thresholding based on skin tone characteristics, and reconstructs the signal while preserving cardiac information.

- **Usage Example:**  
  ```python
  # Initialize the wavelet denoiser with custom parameters
  denoiser = WaveletDenoiser(wavelet='db4', level=3)
  
  # Apply denoising with skin-tone adaptation
  denoised_signal = denoiser.apply_wavelet_denoising(
      signal=ppg_smoothed,
      motion_burst=motion_flags,
      skin_tone='III-IV',
      noise_level=0.2
  )
  ```

- **Parameters & Returns:**

  | Parameter | Type | Description |
  |-----------|------|-------------|
  | wavelet | str | Type of wavelet to use for denoising (default: 'db4') |
  | level | int | Level of wavelet decomposition (default: 3) |
  
  | Method Parameter | Type | Description |
  |------------------|------|-------------|
  | signal | np.ndarray | Input signal to be denoised |
  | motion_burst | np.ndarray | Binary array indicating motion periods |
  | skin_tone | str | Skin tone category (I-II, III-IV, V-VI) |
  | noise_level | float | Estimated noise level (0-1) |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | denoised | np.ndarray | Denoised signal with preserved cardiac components |

> **Workflow Overview:**  
> 1. The denoiser selects appropriate wavelet type and thresholding method based on skin tone:
>    - Lighter skin tones (I-II): db8 wavelet with universal thresholding.
>    - Medium skin tones (III-IV): sym6 wavelet with SURE thresholding.
>    - Darker skin tones (V-VI): coif3 wavelet with Bayesian thresholding.
> 2. Determines the maximum safe decomposition level based on signal length.
> 3. Performs wavelet decomposition using PyWavelets.
> 4. Extracts cardiac component using bandpass filtering for preservation.
> 5. Identifies cardiac peaks to create a cardiac preservation mask.
> 6. Applies frequency-dependent thresholding:
>    - Less aggressive thresholding for lower frequency bands (cardiac).
>    - More aggressive thresholding for higher frequency bands (noise).
>    - Uses skin-tone appropriate thresholding method.
> 7. Reconstructs the signal from thresholded coefficients.
> 8. Blends the denoised signal with:
>    - Original signal to preserve characteristics.
>    - Enhanced cardiac component to boost physiological information.
> 9. Applies dynamic cutoff based on estimated pulse rate.
> 10. Enforces amplitude constraints to prevent signal inflation.

- **Dependencies:**  
  - `pywt`: For wavelet transforms and coefficient thresholding
  - `numpy`: For array operations and mathematical functions
  - `scipy.signal`: For bandpass filtering and peak detection
  - `pandas`: For data handling

### 🔗 SignalProcessingPipeline
- **Purpose:**  
  Orchestrates the complete signal processing workflow by coordinating all component interactions. This pipeline handles end-to-end processing from raw data to cleaned signals, managing memory-efficient chunk processing, device-specific adaptations, and signal quality optimization.

- **Usage Example:**  
  ```python
  # Initialize the pipeline
  pipeline = SignalProcessingPipeline()
  
  # Process an entire dataset
  cleaned_dataset = pipeline.process_signal(raw_dataset)
  
  # Save the cleaned dataset
  pipeline.save_cleaned_dataset(
      dataset=cleaned_dataset,
      output_path="data/cleaned_dataset.parquet"
  )
  ```

- **Parameters & Returns:**

  | Method Parameter | Type | Description |
  |------------------|------|-------------|
  | dataset | pd.DataFrame | Raw unified dataset with PPG and accelerometer data |
  | output_path | str | Path to save the cleaned dataset |
  
  | Returns | Type | Description |
  |---------|------|-------------|
  | dataset | pd.DataFrame | Dataset with cleaned PPG signals (bvp_cleaned, bvp_smoothed, bvp_denoised) |

> **Workflow Overview:**  
> 1. The pipeline initializes all processing components:
>    - AdaptiveFilter for motion artifact removal.
>    - KalmanFilter for temporal smoothing.
>    - WaveletDenoiser for frequency-domain denoising.
>    - MotionArtifactDetector for identifying motion periods.
> 2. During processing, it:
>    - Applies device-specific preprocessing based on device profiles.
>    - Detects motion bursts using the MotionArtifactDetector.
>    - Processes data in chunks to optimize memory usage:
>      - Chunk size of 20,000 samples with 250-sample overlap.
>      - Adapts filter parameters based on noise level.
>      - Applies the complete filtering chain to each chunk.
>    - Performs phase-aware blending between chunks to prevent boundary artifacts.
>    - Applies post-processing for signal enhancement:
>      - Direct signal preservation to maintain original characteristics.
>      - Cardiac component enhancement to boost physiological information.
>      - SNR optimization to maximize signal quality.
>    - Updates smoothed and denoised signals for consistency.
>    - Calculates physiological SNR for quality monitoring.
> 3. The final output includes:
>    - Original signal (bvp).
>    - Cleaned signal (bvp_cleaned).
>    - Smoothed signal (bvp_smoothed).
>    - Denoised signal (bvp_denoised).
>    - Motion burst flags and metadata.

- **Dependencies:**  
  - `pandas`: For DataFrame handling
  - `numpy`: For array operations
  - `scipy.signal`: For filtering and signal processing
  - `fastdtw`: For dynamic time warping
  - `tqdm`: For progress tracking
  - `os`: For file operations
  - `logging`: For diagnostic information
  - All internal components: AdaptiveFilter, KalmanFilter, WaveletDenoiser, MotionArtifactDetector