import numpy as np


class KalmanFilter:
    def __init__(self, initial_state: float = 0.0, 
                 process_noise: float = 1e-2,  # Significantly increased
                 measurement_noise: float = 1e-2):  # Significantly reduced
        """
        Initialize the Kalman filter with enhanced parameters
        
        Parameters:
            initial_state (float): Initial state estimate
            process_noise (float): Process noise covariance (Q)
            measurement_noise (float): Measurement noise covariance (R)
        """
        self.state = initial_state
        self.error_covariance = 1.0  # Significantly increased from 0.5
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.prev_innovations = []
        self.prev_states = []

    def update(self, measurement: float, motion_burst: bool = False) -> float:
        # Motion-aware noise adaptation
        if motion_burst:
            self.process_noise = 2e-1  # Doubled from 1e-1
            measurement_weight = 0.95  # Increased from 0.9
        else:
            self.process_noise = 1e-2  # Doubled from 5e-3
            measurement_weight = 0.99
        
        # Prediction
        predicted_state = self.state
        predicted_error = self.error_covariance + self.process_noise
        
        # Update
        innovation = measurement - predicted_state
        kalman_gain = predicted_error / (predicted_error + self.measurement_noise)
        
        # Virtually unlimited innovation - allow the filter to track large changes
        max_innovation = 1000.0  # Dramatically increased from 100.0
        innovation = np.clip(innovation, -max_innovation, max_innovation)
        
        # Minimal physiological plausibility check
        if len(self.prev_innovations) >= 2:
            innov_threshold = 10.0 * np.nanmedian(np.abs(self.prev_innovations[-5:]))  # Doubled from 5.0
            if abs(innovation) > innov_threshold:
                innovation *= 0.7  # Less aggressive reduction
        else:
            innov_threshold = np.inf
        
        self.state = predicted_state + kalman_gain * innovation * measurement_weight
        self.error_covariance = (1 - kalman_gain) * predicted_error
        
        # Store current values for future reference
        self.prev_innovations.append(innovation)
        self.prev_states.append(self.state)
        
        # Minimal post-update smoothing
        if len(self.prev_states) >= 3:
            state_smooth = 0.98*self.state + 0.02*np.nanmedian(self.prev_states[-5:])  # Less smoothing
        else:
            state_smooth = self.state
        
        # Maintain bounded history
        self.prev_innovations = self.prev_innovations[-50:]  # Reduced from 100
        self.prev_states = self.prev_states[-25:]  # Reduced from 50
        
        return state_smooth  # Removed clipping to allow negative values

    def apply_kalman_filter(self, signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        # Initialize with the first signal value to preserve amplitude characteristics
        self.state = signal[0]
        
        # Pre-calculate signal statistics for preservation
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        filtered = np.zeros_like(signal)
        for i in range(len(signal)):
            filtered[i] = self.update(signal[i], motion_burst[i])
            
            # Minimal physiological constraints
            if i > 10:
                avg = np.mean(filtered[i-5:i])
                # Minimal smoothing
                smooth_factor = 0.1 if motion_burst[i] else 0.05  # Further reduced
                filtered[i] = smooth_factor*filtered[i] + (1-smooth_factor)*avg
                
            # Heavily favor original signal
            filtered[i] = 0.2*filtered[i] + 0.8*signal[i]  # Further increased original signal
        
        # Post-processing to preserve amplitude characteristics
        filtered_mean = np.mean(filtered)
        filtered_std = np.std(filtered)
        
        # Rescale to match original signal statistics
        normalized = (filtered - filtered_mean) / (filtered_std + 1e-9)
        rescaled = normalized * signal_std + signal_mean
        
        return rescaled