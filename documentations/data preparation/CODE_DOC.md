# 📊 Module: Data Preparation

## 1. 📝 Overview

- **Purpose:**  
  Standardizes raw physiological data from heterogeneous datasets (WESAD, PhysioNet) into a unified format for machine learning. Handles sensor alignment, synthetic noise injection, and validation to ensure consistency across device types and skin tones.

- **Scope:**  
  Solves three core challenges:  
  1. **Sensor heterogeneity** – harmonizes accelerometer orientations (NED↔ENU) and PPG wavelength differences.  
  2. **Label inconsistency** – maps dataset-specific labels to a universal stress/affect scheme.  
  3. **Real-world variability** – simulates device-specific noise (Apple Watch vs. Galaxy Watch) and Fitzpatrick skin-tone effects.  
  Acts as the critical bridge between raw sensor data and model-ready inputs.

- **Key Innovations:**  
  - **🎛️ Hierarchical noise simulation:** Device + skin-tone noise layers with physiologically grounded parameters (e.g., 4–8 μV PPG noise scaling).  
  - **⏳ Temporal-spatial alignment:** Combines dynamic time warping for timestamp synchronization with axis remapping for sensor fusion.  
  - **📂 Schema-first output:** Enforces Parquet file standards with embedded metadata (noise levels, device type) for reproducible preprocessing.

---

## 2. 🔗 Context & Integration

- **Position in Project:**  
  Acts as the second layer in the data pipeline, processing raw sensor data from the `data_loading` module and delivering standardized inputs for downstream tasks (e.g., feature extraction, model training). Bridges low-level sensor readings and high-level analytics.

- **Inter-Module Dependencies:**  
  - **Input Dependencies:**  
    - `data_loading` (`PhysioNetLoader`, `WESADLoader`): Ingests raw physiological signals and metadata.  
    - `motion_simulator.py`: Optional dependency for advanced motion artifact generation.  
  - **Output Dependencies:**  
    - `signal_processing`: Receives aligned, noise-augmented data for heartbeat segmentation/feature extraction.  
  - **Internal Dependencies:**  
    - **`sensor_alignment.py`**: Handles critical tasks like accelerometer axis remapping (NED→ENU) and PPG wavelength normalization.  
    - **`noise_simulator.py`**: Generates synthetic noise profiles based on device type (Apple/Galaxy Watch) and Fitzpatrick skin-tone categories.

### 🧩 Workflow Diagram  
```mermaid
flowchart TD
    A[data_loading\n[PhysioNetLoader, WESADLoader]] --> B[Data Preparation Module]
    
    subgraph Data_Preparation_Module
        direction TB
        B --> C[Preprocess Data\n(Clip ACC, Label Mapping)]
        C --> D[SensorAligner]
        D --> E[Align PPG Wavelengths]
        D --> F[Remap ACC Axes\n(NED→ENU)]
        C --> G[NoiseSimulator]
        G --> H[Add Device Noise\n(Apple/Galaxy)]
        G --> I[Add Skin-Tone Effects\n(I-II to V-VI)]
        H --> J[Validate Data\n(Schema, Sampling Rate)]
        I --> J
        J --> K[Save to Parquet\n(Metadata Embedding)]
    end
    
    K --> L[signal_processing\n(Feature Extraction)]
    K --> M[model_training\n(Stress Classification)]
    
    subgraph Orchestration
        N[UnifiedDataPipeline] -->|Batch Process| O[Subjects 1-17]
        N -->|Merge Datasets| P[Cross-Dataset Alignment]
    end
