import numpy as np

class FilterKalman:
    """
    A real-time Kalman filter that estimates the state of a system given noisy measurements.
    """

    def __init__(self, num_taps=100):
        """
        Initializes the Kalman filter with default parameters.
        """
        self.A = np.array([[1, 1], [0, 1]])  # State transition matrix
        self.B = np.array([0.5, 1]).reshape(-1, 1)  # Control matrix
        self.H = np.array([1, 0]).reshape(1, -1)  # Observation matrix
        self.Q = np.eye(2) * 0.001  # Process noise covariance
        self.R = np.eye(1) * 0.1  # Measurement noise covariance
        self.P = np.eye(2)  # Initial estimate error covariance
        self.x = np.array([0, 1])  # Initial state estimate

    def filter(self, y, x):
        """
        Predict the state and update with a new measurement.

        Parameters:
        y (float): Measurement.
        x (float): Control input (if any).

        Returns:
        yhat (float): The estimated noise in y.
        err (float): The error of the sample.
        """
        # Predict
        self.x = np.dot(self.A, self.x) + np.dot(self.B, x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        # Update
        y_pred = np.dot(self.H, self.x)
        yhat = y_pred
        y_residual = y - y_pred
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y_residual)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        err = np.linalg.norm(y_residual)
        return yhat, err

