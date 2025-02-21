import numpy as np

def add_motion_bursts(signal: np.ndarray, 
                     burst_duration: float = 1.5,  # seconds
                     intensity: float = 2.0) -> np.ndarray:
    """
    Simulates sudden arm movements (e.g., gesturing)
    """
    burst_samples = int(burst_duration * 256)  # 256Hz sampling
    burst = intensity * np.random.randn(burst_samples)
    
    # Random insertion
    start_idx = np.random.randint(0, len(signal)-burst_samples)
    signal[start_idx:start_idx+burst_samples] += burst
    
    return signal 