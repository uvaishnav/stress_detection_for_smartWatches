graph TD
A[Raw Data] --> B[WESAD Loader]
A --> C[PhysioNet Loader]
B --> D[Wrist Data Extraction]
B --> E[Label Mapping 0-3→Stress]
C --> F[Exam Period Alignment]
C --> G[Implicit Label Creation]
D --> H[30Hz Resampling]
F --> H
G --> I[Binary Stress Labels]
E --> I
H --> J[Smartwatch Simulation]
J --> K[Unified DataFrame]
K --> L[Validation & Visualization]


I want to know the progress of the phase till now. according to our plan we have discussed, I want to know how much we have done and whats left to do as of now