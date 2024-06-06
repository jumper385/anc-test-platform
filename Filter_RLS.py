import numpy as np
from collections import deque

class FilterRLS:
    """
    A real-time AF that takes incoming x and y samples as they are sampled... then adapts the AF...
    This uses the RLS filter algorithm.
    """

    def __init__(self, num_taps=100, delta=1.0, lam=0.99):
        self.num_taps = num_taps
        self.lam = lam
        self.delta = delta
        self.weights = np.zeros(num_taps)
        self.P = np.eye(num_taps) / delta
        self.x_buff = np.zeros(num_taps)

    def filter(self, x, y):
        """
        For noise canceling applications, you want the target (y) to be the noisy signal and the input noise to be the noise reference.
        This is because we're estimating the amplitude of the noise reference within the noise signal NOT extracting the noise signal from the noise.

        Parameters:
        x (float): Sample of noise reference
        y (float): Sample of the noisy signal

        Returns:
        yhat (float): The estimated noise in y
        err (float): The error of the sample
        """
        # Shift the buffer and insert the new sample
        self.x_buff = np.roll(self.x_buff, -1)
        self.x_buff[-1] = x

        # Calculate the output
        yhat = np.dot(self.weights, self.x_buff)

        # Update the filter
        err = y - yhat
        self.update(x, y, yhat, err)

        return yhat, err

    def update(self, x, y, yhat, err):
        """
        Update the filter weights using the RLS algorithm.

        Parameters:
        x (float): Sample of noise reference
        y (float): Sample of the noisy signal
        yhat (float): The estimated noise in y
        err (float): The error of the sample
        """
        x_buff = self.x_buff.reshape(-1, 1)
        Pi = np.dot(self.P, x_buff)
        k = Pi / (self.lam + np.dot(x_buff.T, Pi))
        self.weights += (err * k.flatten())
        self.P = (self.P - np.dot(k, Pi.T)) / self.lam
