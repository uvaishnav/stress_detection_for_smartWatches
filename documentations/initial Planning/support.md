## Project Context Transfer

### Current Phase: [Phase Name]

**Project State:**
- Data Processing: [Completed steps]
- Model Architecture: [Current structure]
- Evaluation Metrics: [Latest results]
- Deployment Status: [Current stage]

**Key Decisions:**
1. [Decision 1 with rationale]
2. [Decision 2 with constraints]

**Pending Challenges:**
- [Challenge 1]
- [Challenge 2]

**Code Snippets:**
```python
# Core processing function
def process_signals(ppg, accel):
    {{ current_implementation }}
```

**References:**
- Paper1: [Usage in Phase 2]
- Dataset1: [Preprocessing approach]

### Phase N Completion Report
**Accomplished:**
- Implemented motion artifact removal
- Achieved 82% validation accuracy

**Artifacts:**
- signal_processing.py v1.2
- Model checkpoint: stressnet_phase3.pth

**Next Phase Needs:**
- Help optimizing model for Apple Watch
- Guidance on CoreML conversion

New Chat Message:
[Paste Phase Report]
[Paste Context Transfer Template]
Current Task: [Phase N+1 Objective]


2. Context Snapshot

{
    "phase": 3,
    "artifacts": ["processed_signals.pkl", "eda_proxy.py"],
    "metrics": {"snr": 18.2, "rmse": 0.15},
    "next_phase": "Feature Engineering"
}

Provide these papers for Phase N:
- Paper1: [Title] (Key Technique)
- Paper2: [Title] (Relevant Method)

Current Phase Context:
- Previous Phase Output: [File Path]
- Immediate Technical Challenge: [Description]


Phase 1: Setup and Data Loading. This includes environment setup, installing necessary tools, acquiring datasets, and initial data exploration. Tools like Conda, Python, and libraries such as Pandas and Matplotlib are essential here. Success would be visualizing raw data.

Phase 2: Data Preparation. Here, aligning signals from different sensors, handling missing data, and augmenting data to simulate smartwatch limitations. Techniques like resampling and synthetic noise addition are key. The success is having clean, aligned data.

in phase 2 : addtional resources used : 
Use 0.42μV (Apple) / 0.38μV (Samsung) from "Wearable PPG Noise Analysis" (2023)
Add melanin absorption coefficients from "Optical Skin Effects" (J. Biomed. Opt. 2020)
"Motion Artifact Reduction in PPG Signals" (IEEE TBME 2015)
"Skin Tone Effects on PPG Signals" (Nature Sci. Reports 2021)

Phase 3: Signal Processing. Focus on cleaning PPG signals, motion artifact removal, and deriving proxy measures for missing sensors like EDA. Using filters and wavelet transforms. Success is improved signal quality metrics.

Phase 4: Feature Engineering. Extracting time-domain, frequency-domain, and multimodal features. Using libraries like TSFRESH and creating images for CNNs. Success is a robust feature set with high importance scores.

Phase 5: Model Development. Building baseline models, then advanced architectures like CNN-LSTM hybrids. Hyperparameter tuning and cross-validation. Success is achieving target accuracy metrics.

Phase 6: Model Evaluation. Rigorous testing with multiple metrics, real-time simulation, and bias checks. Tools like scikit-learn for metrics and SHAP for explainability. Success is meeting all benchmark metrics.

Phase 7: Deployment Preparation. Optimizing the model through pruning and quantization, converting to edge formats, and building a real-time pipeline. Success is a deployable model meeting size and latency constraints.

Phase 8: Maintenance. Implementing monitoring, feedback loops, and continuous learning. Success is maintaining performance over time with updates.


First, for the Project Setup & Data Loading phase, the main papers would be the ones that used datasets like WESAD. The WESAD paper by Schmidt et al. is crucial here. Also, the Choi et al. paper on ambulatory stress monitoring would be relevant for understanding real-world data challenges.

Next, Data Preparation. The Mental Stress Prediction paper by Verma and Singh might discuss data handling from wrist devices. The Quadrini et al. paper on encoding signals could be useful for simulating smartwatch data.

Signal Processing phase needs motion artifact removal techniques. Choi et al. again, since they worked on real-world data. The Quadrini paper's methods for handling sensor limitations could apply here too.

Feature Engineering would benefit from Quadrini's encoding methods (GASF, MTF) and the Mental Stress Prediction paper's ML approaches. The Rashid et al. paper on context-aware fusion might have feature fusion strategies.

Model Development phase should reference Rashid's multi-scale attention and causal graphs, Quadrini's CNN approaches, and Choi's traditional vs. deep learning comparison.

Evaluation phase needs validation methods from Choi and Rashid, and real-world testing from the Apple Watch HRV validation study.

Deployment Prep requires optimization techniques from Rashid and the Apple Watch study's constraints. Quadrini's efficiency considerations would also be relevant.

Maintenance phase can use Rashid's continuous learning and the Apple Watch study's longitudinal aspects.


