## 📚 5. Detailed Component Documentation

### 🔧 5.1 BasePreparer Class

- **Purpose:**  
  Abstract base class implementing core data standardization logic - label unification, sensor validation, and schema enforcement across datasets.

- **Usage Example:**  
```python
preparer = BasePreparer(data_path="/raw/wesad", output_dir="processed/")
result = preparer.process_subject(subject_id=15)
```

| 🛠️ Parameter | 🔧 Type | 📝 Description |
|---------------|----------|----------------|
| `data_path`   | `str`    | Path to raw dataset directory |
| `output_dir`  | `str`    | Output directory for processed files |

| 🎯 Returns   | 🔧 Type | 📝 Description |
|---------------|----------|----------------|
| `result`      | `dict`   | Processing status with saved file paths |

**⚙️ Internal Workflow:**  
1. Loads raw data via `load_subject()`  
2. Clips accelerometer values to ±3.5g (prevents sensor saturation)  
3. Aligns sensors using `SensorAligner`  
4. Maps labels to unified scheme via `map_labels()`  
5. Validates sampling rate (30Hz ±1%)  
6. Generates noise variants with `NoiseSimulator`  
7. Saves Parquet files with embedded metadata  

**🔗 Dependencies:**  
- `pandas`, `numpy` (data manipulation)  
- `pathlib` (path handling)  
- `SensorAligner` (axis correction)  
- `NoiseSimulator` (augmentation)  

---

### 🔧 5.2 NoiseSimulator.add_device_noise()

- **Purpose:**  
  Injects physiologically realistic noise into PPG/ACC signals based on device hardware and skin tone characteristics.

- **Usage Example:**  
```python
simulator = NoiseSimulator()
noisy_df = simulator.add_device_noise(clean_df, device='apple_watch', skin_tone='V-VI')
```

| 🛠️ Parameter  | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `data`         | `DataFrame`   | Clean sensor data with BVP/ACC columns |
| `device`       | `str`         | Target device profile ('apple_watch'/'galaxy_watch') |
| `skin_tone`    | `str`         | Fitzpatrick skin category (I-II to V-VI) |

| 🎯 Returns    | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `noisy_df`     | `DataFrame`   | Augmented data with device/skin metadata |

**⚙️ Internal Workflow:**  
1. Preserves original timestamps and length  
2. Applies device-specific noise:  
   - Apple Watch: AC component gain + Gaussian noise  
   - Galaxy Watch: Low-frequency drift + Laplace noise  
3. Adds motion artifacts proportional to ACC magnitude  
4. Restores original values where NaNs introduced  

**🔗 Dependencies:**  
- `numpy.random` (noise generation)  
- `pandas.DataFrame` (signal manipulation)  

---

### 🔧 5.3 SensorAligner.align_acc()

- **Purpose:**  
  Converts accelerometer axes to ENU (East-North-Up) standard across datasets.

- **Usage Example:**  
```python
aligner = SensorAligner()
aligned_df = aligner.align_acc(raw_df, dataset='physionet')
```

| 🛠️ Parameter  | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `data`         | `DataFrame`   | Raw accelerometer data |
| `dataset`      | `str`         | Source dataset identifier |

| 🎯 Returns    | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `aligned_df`   | `DataFrame`   | Axis-remapped ACC data |

**⚙️ Internal Workflow:**  
1. Loads dataset-specific remapping rules:  
   - PhysioNet: NED→ENU via {x→y, y→x, z→-z}  
   - WESAD: No change (already ENU)  
2. Creates new columns using transformation rules  
3. Drops original ACC columns  

**🔗 Dependencies:**  
- `ACC_REMAPPING` config dictionary  
- `pandas.DataFrame` operations  

---

### 🔧 5.4 UnifiedDataPipeline.process_subject()

- **Purpose:**  
  Coordinates end-to-end processing of a subject across multiple datasets with quality control.

- **Usage Example:**  
```python
pipeline = UnifiedDataPipeline()
results = pipeline.process_subject(101, datasets=['physionet','wesad'])
```

| 🛠️ Parameter  | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `subject_id`   | `int`         | Numeric subject identifier |
| `datasets`     | `list[str]`   | Datasets to process (e.g., ['physionet']) |

| 🎯 Returns    | 🔧 Type       | 📝 Description |
|----------------|---------------|----------------|
| `results`      | `dict`        | Processing status with output file paths |

**⚙️ Internal Workflow:**  
1. Validates subject ID against dataset ranges  
2. Initializes dataset-specific preparers  
3. Executes preprocessing pipeline:  
   - PhysioNet: Processes per exam type (Midterm 1, etc.)  
   - WESAD: Single processing pass  
4. Merges outputs using temporal alignment  
5. Generates unified Parquet file  

**🔗 Dependencies:**  
- `PhysioNetPreparer` / `WESADPreparer`  
- `SensorAligner.temporal_align()`  
- `pandas.concat` (data merging)  

---

> **💡 Key Takeaways:**
> - The **BasePreparer** class is the backbone of data standardization, ensuring consistent preprocessing across datasets.
> - **NoiseSimulator** adds realistic noise profiles, enhancing the robustness of machine learning models.
> - **SensorAligner** ensures uniformity in accelerometer axis orientation, crucial for cross-dataset compatibility.
> - **UnifiedDataPipeline** orchestrates the entire preprocessing workflow, enabling seamless integration of multiple datasets.

### 🔧 5.5 PhysioNetPreparer.process_subject()

- **Purpose:**  
  Processes PhysioNet subjects with exam-type-specific handling, including tag validation and temporal alignment.

- **Usage Example:**  
```python
preparer = PhysioNetPreparer(data_path="/raw/physionet")
result = preparer.process_subject(subject_id=5, exam_type="Midterm 1")
```

| 🛠️ Parameter   | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `subject_id`     | `int`    | Subject ID (1-10 per PhysioNet constraints) |
| `exam_type`      | `str`    | Assessment phase ("Midterm 1", "Final", etc.) |

| 🎯 Returns     | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `result`         | `dict`   | Contains saved file paths and processing metadata |

**⚙️ Internal Workflow:**  
1. Checks tags.csv validity via `loader.is_tags_empty()`  
2. Generates UTC timestamps using recording start times  
3. Resamples to 30Hz using reference index alignment  
4. Applies PhysioNet-specific ACC remapping (NED→ENU)  
5. Saves clean/noisy variants with exam-type in filenames  

**🔗 Dependencies:**  
- `PhysioNetLoader` (raw data ingestion)  
- `SensorAligner._add_physionet_timestamps()`  
- `NoiseSimulator.device_profiles`  

---

### 🔧 5.6 SensorAligner.cross_dataset_align()

- **Purpose:**  
  Final normalization for merged datasets, adjusting PPG gain by skin tone and calibrating device-specific ACC.

- **Usage Example:**  
```python
aligner = SensorAligner()
merged_df = aligner.cross_dataset_align(combined_data)
```

| 🛠️ Parameter   | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `merged`         | `DataFrame`   | Combined data from multiple sources |

| 🎯 Returns     | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `merged`         | `DataFrame`   | Harmonized data with unified PPG/ACC scales |

**⚙️ Internal Workflow:**  
1. Applies skin-tone gain factors (I-II: 1.0, V-VI: 0.8) to BVP  
2. Calibrates Apple Watch ACC_X (+5% scaling)  
3. Converts all timestamps to UTC with millisecond precision  

**🔗 Dependencies:**  
- Williams et al. (2022) skin-tone normalization framework  
- `pandas.Timestamp.tz_convert()`  

---

### 🔧 5.7 motion_simulator.add_motion_bursts()

- **Purpose:**  
  Injects short-duration high-intensity noise to simulate sudden arm movements (e.g., gesturing).

- **Usage Example:**  
```python
noisy_signal = add_motion_bursts(clean_signal, burst_duration=2.0, intensity=3.0)
```

| 🛠️ Parameter   | 🔧 Type     | 📝 Description |
|------------------|-------------|----------------|
| `signal`         | `ndarray`   | 1D signal array |
| `burst_duration` | `float`     | Duration in seconds (default=1.5) |
| `intensity`      | `float`     | Noise magnitude multiplier (default=2.0) |

| 🎯 Returns     | 🔧 Type     | 📝 Description |
|------------------|-------------|----------------|
| `noisy_signal`   | `ndarray`   | Signal with injected motion artifacts |

**⚙️ Internal Workflow:**  
1. Calculates burst length: `samples = duration × 256Hz`  
2. Generates Gaussian noise burst scaled by intensity  
3. Inserts burst at random position in signal  

**🔗 Dependencies:**  
- `numpy.random.randn()`  
- Mannini et al. (2016) movement characterization  

---

### 🔧 5.8 UnifiedDataPipeline._merge_files()

- **Purpose:**  
  Combines processed Parquet files from multiple datasets into a unified DataFrame with schema enforcement.

- **Usage Example:**  
```python
pipeline = UnifiedDataPipeline()
merged_df = pipeline._merge_files(dataset_results)
```

| 🛠️ Parameter   | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `dataset_results`| `dict`   | Dictionary of processed file paths |

| 🎯 Returns     | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `merged`         | `DataFrame`   | Combined dataset with UTC-indexed data |

**⚙️ Internal Workflow:**  
1. Loads all Parquet files into memory  
2. Filters columns to required schema  
3. Converts timestamps to UTC timezone  
4. Sorts merged data by temporal index  
5. Validates against final column requirements  

**🔗 Dependencies:**  
- `pandas.concat` (axis=0 merging)  
- Vest et al. (2021) resampling best practices  

---

> **💡 Key Takeaways:**
> - **PhysioNetPreparer** ensures that data from the PhysioNet dataset is processed with specific handling for different exam types, ensuring temporal alignment and proper tagging.
> - **SensorAligner.cross_dataset_align** harmonizes data across multiple datasets by normalizing PPG signals based on skin tone and calibrating ACC signals for device-specific differences.
> - **motion_simulator.add_motion_bursts** introduces realistic motion artifacts into signals, simulating sudden arm movements, which is crucial for robust model training.
> - **UnifiedDataPipeline._merge_files** consolidates data from multiple sources into a single, unified DataFrame, ensuring consistent formatting and schema enforcement.


### 🔧 5.9 WESADPreparer.process_subject()

- **Purpose:**  
  Processes WESAD subjects with dataset-specific validation, including BVP range checks and meditation sample thresholds.

- **Usage Example:**  
```python
preparer = WESADPreparer(data_path="/raw/wesad")
result = preparer.process_subject(subject_id=10)
```

| 🛠️ Parameter   | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `subject_id`     | `int`    | Subject ID (2-17 per WESAD constraints) |

| 🎯 Returns     | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `result`         | `dict`   | Contains Parquet paths for clean/noisy variants |

**⚙️ Internal Workflow:**  
1. Loads raw WESAD data via `WESADLoader`  
2. Generates synthetic timestamps at 30Hz (33.333ms intervals)  
3. Validates BVP range (-2.5 to +2.5 volts)  
4. Checks meditation class distribution (≥5% samples)  
5. Clips ACC values to ±3.5g  
6. Generates device/skin-tone noise variants  

**🔗 Dependencies:**  
- `WESADLoader` (SMILEbox data structure parsing)  
- `NoiseSimulator.add_device_noise()`  
- `pandas.date_range` (timestamp generation)  

---

### 🔧 5.10 BasePreparer.validate_output()

- **Purpose:**  
  Enforces post-processing data integrity through label validity and temporal monotonicity checks.

- **Usage Example:**  
```python
is_valid = preparer.validate_output(processed_df)
```

| 🛠️ Parameter   | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `data`           | `DataFrame`   | Processed data to validate |

| 🎯 Returns     | 🔧 Type | 📝 Description |
|------------------|----------|----------------|
| `is_valid`       | `bool`   | True if all validation checks pass |

**⚙️ Internal Workflow:**  
1. Verifies labels match `TARGET_LABELS` keys (0-3)  
2. Checks timestamp monotonicity (no time travel)  
3. PhysioNet-specific: Validates stress label ratio <30%  
4. WESAD-specific: Confirms BVP values within ±2.5V  

**🔗 Dependencies:**  
- Garcia-Ceja et al. (2019) quality metrics  
- `pandas.Index.is_monotonic_increasing`  

---

### 🔧 5.11 SensorAligner.temporal_align()

- **Purpose:**  
  Aligns timestamps across datasets using time-warping interpolation for synchronized analysis.

- **Usage Example:**  
```python
aligned_df = aligner.temporal_align(source_df, reference_df)
```

| 🛠️ Parameter   | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `data`           | `DataFrame`   | Source data to align |
| `reference`      | `DataFrame`   | Target timestamp index |

| 🎯 Returns     | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `aligned`        | `DataFrame`   | Data reindexed to reference timeline |

**⚙️ Internal Workflow:**  
1. Uses `reindex_like()` to match reference index  
2. Applies time-based linear interpolation  
3. Forward-fills label values across NaN gaps  

**🔗 Dependencies:**  
- Schmidt et al. (2018) multimodal fusion strategy  
- `pandas.DataFrame.reindex()`  

---

### 🔧 5.12 NoiseSimulator._apply_ppg_noise()

- **Purpose:**  
  Internal method implementing PPG noise injection logic with device-specific profiles.

- **Usage Example:**  
```python
# Called internally by add_device_noise()
noisy_ppg = self._apply_ppg_noise(clean_df, 'apple_watch', 'III-IV')
```

| 🛠️ Parameter   | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `data`           | `DataFrame`   | Clean PPG signal data |
| `device`         | `str`         | Target device profile |
| `skin_tone`      | `str`         | Fitzpatrick skin category |

| 🎯 Returns     | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `data`           | `DataFrame`   | PPG signal with applied noise |

**⚙️ Internal Workflow:**  
1. Separates DC/AC components (Apple Watch)  
2. Applies skin-tone gain (0.6-1.0) to AC component  
3. Adds Gaussian/Laplace noise based on device  
4. For Galaxy Watch: Introduces linear drift  

**🔗 Dependencies:**  
- Karlen et al. (2013) thermal drift model  
- `numpy.random.normal/laplace`  

---

### 🔧 5.13 BasePreparer.save_processed()

- **Purpose:**  
  Finalizes processed data with schema enforcement before Parquet serialization.

- **Usage Example:**  
```python
preparer.save_processed(df, "subject_15_clean")
```

| 🛠️ Parameter   | 🔧 Type       | 📝 Description |
|------------------|---------------|----------------|
| `df`             | `DataFrame`   | Processed data to save |
| `filename`       | `str`         | Base filename template |

**⚙️ Internal Workflow:**  
1. Adds missing columns with defaults (`noise_level=0.0`)  
2. Drops non-standard columns  
3. Enforces dtype constraints (e.g., `label=int8`)  
4. Applies filename template: `{device}_{skin_tone}_{filename}`  
5. Writes to Parquet with millisecond timestamp precision  

**🔗 Dependencies:**  
- Apache Parquet schema requirements  
- `pandas.astype` (type enforcement)  

---

### 🔧 5.14 UnifiedDataPipeline.run_batch()

- **Purpose:**  
  Executes large-scale processing of multiple subjects across datasets with automated merging.

- **Usage Example:**  
```python
pipeline.run_batch(subject_ids=[5,6,7], datasets=['physionet','wesad'])
```

| 🛠️ Parameter   | 🔧 Type         | 📝 Description |
|------------------|-----------------|----------------|
| `subject_ids`    | `list[int]`     | Subjects to process |
| `datasets`       | `list[str]`     | Target datasets |

**⚙️ Internal Workflow:**  
1. Filters valid subject-dataset combinations  
2. Parallelizes `process_subject()` calls  
3. Merges outputs into unified dataset  
4. Generates global Parquet file with UTC-sorted data  

**🔗 Dependencies:**  
- Vest et al. (2021) batch processing framework  
- `Concurrent.futures` (implicit parallelization)  

---

> **💡 Key Takeaways:**
> - **WESADPreparer** ensures that WESAD data is processed with specific validations, such as BVP range checks and meditation sample thresholds, ensuring high-quality data output.
> - **BasePreparer.validate_output** enforces strict data integrity checks, ensuring that the processed data adheres to predefined standards.
> - **SensorAligner.temporal_align** synchronizes timestamps across datasets using interpolation, enabling seamless cross-dataset analysis.
> - **NoiseSimulator._apply_ppg_noise** injects realistic PPG noise based on device-specific profiles, enhancing the robustness of the data.
> - **BasePreparer.save_processed** finalizes the data with schema enforcement and saves it in Parquet format, ensuring consistency and compatibility.
> - **UnifiedDataPipeline.run_batch** automates the processing of multiple subjects across datasets, leveraging parallelization for efficiency.


