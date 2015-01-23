from hindemith.types.hmarray import hmarray
import numpy as np


class Blob(object):
    def __init__(self, num, channels, height, width):
        self.capacity = 0
        self.reshape(num, channels, height, width)

    def reshape(self, num, channels, height, width):
        assert (num, channels, height, width) > (0, 0, 0, 0)
        self.num = num
        self.channels = channels
        self.height = height
        self.width = width
        self.count = num * channels * height * width
        if self.count > self.capacity:
            self.capacity = self.count
            self.data = hmarray(self.capacity, np.float32)
            self.diff = hmarray(self.capacity, np.float32)

    def fill(self, constant):
        self.data.fill(constant)
