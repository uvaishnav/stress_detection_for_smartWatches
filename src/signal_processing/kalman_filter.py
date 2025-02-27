import numpy as np

class KalmanFilter:
    def __init__(self, initial_state: float = 0.0, process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        """
        Initialize the Kalman filter.
        
        Parameters:
            initial_state (float): Initial state estimate.
            process_noise (float): Estimated process noise covariance.
            measurement_noise (float): Estimated measurement noise covariance.
        """
        self.state = initial_state
        self.error_covariance = 1e-3  # Small initial value
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def update(self, measurement: float, motion_burst: bool = False) -> float:
        """
        Update the state estimate based on a new measurement.
        
        Parameters:
            measurement (float): New measurement value.
            motion_burst (bool): Indicates if the current sample is during a motion burst.
        
        Returns:
            float: Updated state estimate.
        """
        if np.isnan(measurement):
            return self.state  # Skip updates for NaN measurements

        adjusted_measurement_noise = self.measurement_noise * (1 + 2 * int(motion_burst))

        predicted_state = self.state
        predicted_error_covariance = self.error_covariance + self.process_noise

        kalman_gain = predicted_error_covariance / (predicted_error_covariance + adjusted_measurement_noise)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance

        return self.state

    def apply_kalman_filter(self, signal: np.ndarray, motion_burst: np.ndarray) -> np.ndarray:
        """
        Apply the Kalman filter to smooth the input signal.
        
        Parameters:
            signal (np.ndarray): Input signal to smooth.
            motion_burst (np.ndarray): Binary array indicating motion bursts (0 or 1).
        
        Returns:
            np.ndarray: Smoothed signal.
        """
        smoothed_signal = np.zeros_like(signal)
        for i, measurement in enumerate(signal):
            smoothed_signal[i] = self.update(measurement, motion_burst=motion_burst[i])
        return smoothed_signal
