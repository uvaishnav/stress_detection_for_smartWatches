# 📊 Module: Data Loading Module

## 1. 📋 Overview

- **Purpose:**  
  Provides standardized interfaces to load, preprocess, and validate physiological sensor data from diverse datasets. Serves as the foundational data ingestion layer for stress detection systems, ensuring consistency in sampling rates, sensor alignment, and data integrity.

- **Scope:**  
  Solves dataset heterogeneity by:
  - 🔄 Unifying sensor data (BVP, ACC, HR, etc.) into a common DataFrame format.
  - ⏱️ Resampling data to a configurable target rate (default: 30 Hz) with anti-aliasing.
  - ✅ Validating sensor ranges and temporal consistency.
  - 🗂️ Handling dataset-specific metadata, event markers, and label semantics.

- **Key Innovations:**  
  - **🛠️ Abstract Base Class Design**: Enforces consistency across datasets via mandatory methods (`load_subject()`, `get_labels()`).
  - **🔇 Anti-Aliasing During Resampling**: Uses Butterworth low-pass filtering to prevent signal distortion during downsampling.
  - **⚖️ Adaptive Validation Rules**: Dataset-specific thresholds (e.g., BVP range: `-2.5–2.5` for WESAD vs. `-1.0–2.0` for PhysioNet) via `VALIDATION_CONFIG`.
  - **🌍 Time-Zone-Aware Indexing**: Ensures timestamps are UTC-normalized for multi-subject alignment.
  - **📂 Robust File Handling**: Supports both CSV (PhysioNet) and compressed pickle (WESAD) formats with graceful error recovery.


Here is your enhanced markdown for the section "Context & Integration" with added emojis and improved readability:

---

## 🧩 **2. Context & Integration**

- **📍 Position in Project:**  
  Serves as the **first layer** in the stress detection pipeline. This module ingests raw physiological signals (e.g., BVP, ACC) from wearable devices or datasets, performs standardization/validation, and outputs cleaned, time-aligned data for downstream analysis. Processed DataFrames are typically passed to feature extraction or machine learning modules.

- **🔗 Inter-Module Dependencies:**  
  - **Input Dependencies:**  
    - Raw dataset files (PhysioNet CSV/WESAD pickle).  
  - **Internal Dependencies:**  
    - Inherits core logic from `base_loader.py` (abstract methods, resampling).  
    - Uses `scipy.signal` for filtering and `pandas` for time-series alignment.  

- **Workflow Diagram:**  
```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart TB
    %% External Inputs
    Input1([PhysioNet Dataset -- CSV Files]):::input
    Input2([WESAD Dataset -- Pickle Files]):::input

    %% Data Loading Module
    subgraph Data_Loading_Module[Data Loading Module]
        direction TB
        BaseLoader[[BaseDataLoader -- Abstract Class]] --> PhysioNetLoader
        BaseLoader --> WESADLoader
        
        subgraph PhysioNetLoader[PhysioNetLoader]
            direction LR
            P1[Load CSV Files] --> P2[Resample to 30Hz]
            P2 --> P3[Apply Anti-Aliasing Filter]
            P3 --> P4[Validate Ranges]
            P4 --> P5[Clean Data]
        end

        subgraph WESADLoader[WESADLoader]
            direction LR
            W1[Load Pickle Files] --> W2[Resample to 30Hz]
            W2 --> W3[Apply Anti-Aliasing Filter]
            W3 --> W4[Validate Labels]
            W4 --> W5[Normalize BVP]
        end
    end

    %% Dependencies
    BaseLoader -.->|Uses| Pandas[[pandas]]
    BaseLoader -.->|Uses| Scipy[[scipy.signal]]

    %% Outputs
    P5 --> Output1[[Physionet Data - Dataframe]]  
    W5 --> Output2[[WESAD data - Dataframe]]  

    %% Styling
    classDef input fill:#d4e8f7,stroke:#2a4c6b
    classDef module fill:#f0f4c3,stroke:#827717
    classDef output fill:#c8e6c9,stroke:#2e7d32
    classDef lib fill:#ffccbc,stroke:#bf360c
    class Pandas,Scipy lib
    class Data_Loading_Module module
    class Output1,Output2 output
```