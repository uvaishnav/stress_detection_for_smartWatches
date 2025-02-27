import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

class AdaptiveFilter:
    """ACC-guided motion artifact removal using LMS algorithm"""
    
    def __init__(self, filter_order=7, step_size=0.005):
        self.filter_order = filter_order
        self.step_size = step_size
        self.fir_coeff = np.zeros(filter_order)
        
    def _compute_reference(self, acc_data: np.ndarray) -> np.ndarray:
        """Create motion reference signal from ACC"""
        # 1. Compute magnitude
        acc_mag = np.linalg.norm(acc_data, axis=1)
        
        # 2. Bandpass filter 0.5-3Hz (step 3 in workflow)
        sos = signal.butter(2, [0.5, 3], 'bandpass', fs=30, output='sos')
        return signal.sosfilt(sos, acc_mag)
    
    def apply_lms(self, ppg: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """Adaptive noise cancellation with dynamic filter order"""
        # Dynamically adjust filter order for short signals
        effective_order = min(self.filter_order, len(ppg) - 1)
        if effective_order < 1:
            return ppg  # Return original if too short
        
        # Initialize coefficients for current order
        fir_coeff = np.zeros(effective_order)
        
        # Normalization and reference computation remains same
        ppg_norm = MinMaxScaler().fit_transform(ppg.reshape(-1,1)).flatten()
        ref = self._compute_reference(acc)
        ref_norm = MinMaxScaler().fit_transform(ref.reshape(-1,1)).flatten()
        
        # LMS implementation with dynamic order
        cleaned = np.zeros_like(ppg_norm)
        for n in range(effective_order, len(ppg_norm)):
            x = ref_norm[n - effective_order : n]
            y = ppg_norm[n] + np.dot(fir_coeff, x)
            fir_coeff += self.step_size * y * x
            cleaned[n] = y
            
        return cleaned * (ppg.max() - ppg.min()) + ppg.mean()
