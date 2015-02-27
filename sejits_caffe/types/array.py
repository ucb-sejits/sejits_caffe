import numpy as np


class Array(np.ndarray):
    @staticmethod
    def zeros(*args, **kwargs):
        return np.zeros(*args, **kwargs).view(Array)

    @staticmethod
    def rand(*args, **kwargs):
        return np.random.rand(*args, **kwargs).view(Array)

    @staticmethod
    def standard_normal(*args, **kwargs):
        return np.random.standard_normal(*args, **kwargs).view(Array)
