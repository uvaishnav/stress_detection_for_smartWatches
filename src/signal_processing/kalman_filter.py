import numpy as np

class KalmanFilter:
    def __init__(self, initial_state: float = 0.0, 
                 process_noise: float = 1e-3,  # Increased from 1e-5
                 measurement_noise: float = 1e-1):  # Increased from 1e-2
        """
        Initialize the Kalman filter with enhanced parameters
        
        Parameters:
            initial_state (float): Initial state estimate
            process_noise (float): Process noise covariance (Q)
            measurement_noise (float): Measurement noise covariance (R)
        """
        self.state = initial_state if initial_state != 0 else 0.01  # Prevent zero lock
        self.error_covariance = 0.1  # Reduced from 1.0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.prev_innovations = []
        self.prev_states = []

    def update(self, measurement: float, motion_burst: bool = False) -> float:
        # Motion-aware noise adaptation
        if motion_burst:
            self.process_noise = 3e-2  # From 2e-2
            measurement_weight = 0.5  # From 0.45
        else:
            self.process_noise = 8e-4  # From 5e-4
            measurement_weight = 0.95  # From 0.9
        
        # Prediction
        predicted_state = self.state
        predicted_error = self.error_covariance + self.process_noise
        
        # Update
        innovation = measurement - predicted_state
        kalman_gain = predicted_error / (predicted_error + self.measurement_noise)
        
        # HR-adaptive innovation limits
        max_innovation = 35.0 * (1 + 0.9*motion_burst)  # From 30.0 * (1 + 0.8)
        innovation = np.clip(innovation, -max_innovation, max_innovation)
        
        # Physiological plausibility check with empty list handling
        if len(self.prev_innovations) >= 2:  # Require minimum 2 samples
            innov_threshold = 2.0 * np.nanmedian(np.abs(self.prev_innovations[-10:]))
            if abs(innovation) > innov_threshold:
                innovation *= 0.3
        else:
            innov_threshold = np.inf  # No thresholding until sufficient history
        
        self.state = predicted_state + kalman_gain * innovation * measurement_weight
        self.error_covariance = (1 - kalman_gain) * predicted_error
        
        # Post-update smoothing with empty list handling
        if len(self.prev_states) >= 3:  # Minimum 3 samples for median
            state_smooth = 0.9*self.state + 0.1*np.nanmedian(self.prev_states[-5:])
        else:
            state_smooth = self.state
        
        # Maintain bounded history
        self.prev_innovations = self.prev_innovations[-100:]  # Keep last 100 innovations
        self.prev_states = self.prev_states[-50:]  # Keep last 50 states
        
        # Reverse smoothing factors
        smooth_factor = 0.85 if motion_burst else 0.6  # From 0.9/0.7
        
        return np.clip(state_smooth, 0, None)

    def apply_kalman_filter(self, signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        filtered = np.zeros_like(signal)
        for i in range(len(signal)):
            filtered[i] = self.update(signal[i], motion_burst[i])
            # Maintain physiological constraints
            if i > 10:
                avg = np.mean(filtered[i-5:i])
                # Dynamic smoothing based on signal quality
                smooth_factor = 0.85 if motion_burst[i] else 0.6  # From 0.9/0.7
                filtered[i] = smooth_factor*filtered[i] + (1-smooth_factor)*avg
        return filtered