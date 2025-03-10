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

    def update(self, measurement: float, motion_burst: bool = False) -> float:
        """
        Update the Kalman filter state with motion-aware adjustments
        
        Parameters:
            measurement (float): New measurement value
            motion_burst (bool): Indicates motion artifact presence
            
        Returns:
            float: Updated state estimate
        """
        if np.isnan(measurement):
            return self.state  # Maintain current state for invalid measurements

        # Adjust measurement noise during motion bursts
        adjusted_measurement_noise = self.measurement_noise * (1 + 2 * int(motion_burst))

        # Prediction step
        predicted_state = self.state
        predicted_error_covariance = self.error_covariance + self.process_noise

        # Update step
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + adjusted_measurement_noise)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance

        return self.state

    def apply_kalman_filter(self, signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        """
        Apply the Kalman filter to an entire signal with motion awareness
        
        Parameters:
            signal (np.ndarray): Input signal to filter
            motion_burst (np.ndarray): Motion artifact indicators
            
        Returns:
            np.ndarray: Filtered signal
        """
        smoothed_signal = np.zeros_like(signal)
        self.state = signal[0] if len(signal) > 0 else 0.01
        self.error_covariance = 0.1
        
        for i, measurement in enumerate(signal):
            if motion_burst[i]:
                # Modified: Partial reset instead of full measurement adoption
                self.state = 0.7 * self.state + 0.3 * measurement
                self.error_covariance *= 1.2
            smoothed_signal[i] = self.update(measurement, motion_burst[i])
        return smoothed_signal