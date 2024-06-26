import numpy as np
from collections import deque

class FilterLMS:
    def __init__(self, num_taps=1024, lr=0.4):
        self.learning_rate = lr
        self.num_taps = num_taps
        self.weights = np.zeros(num_taps)
        self.output = 0
        self.x_buff = np.zeros(num_taps)

    def filter(self, x, y):
        """
        For noise canceling applications, you want the target (y) to be the noisy signal and the input noise to be the noise reference. This is because, we're estimateing the amplitude of the noise reference within the noise signal NOT extracting the noise signal from the noise"

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

        if (np.abs(yhat) > 2**32 * 100 ):
            yhat = yhat/np.abs(yhat) * 2**32

        err = self.update(x, y, yhat)
        return yhat, err

    def update(self, x, y, yhat):
        e = y - yhat

        if (np.abs(x) > 0):
            nlms_step = (self.learning_rate * e * self.x_buff) / np.dot(self.x_buff, self.x_buff)
            self.weights += nlms_step

        return e
