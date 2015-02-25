import unittest
import numpy as np
from scipy.ndimage.filters import convolve
from hindemith import hmarray

from sejits_caffe.operations.convolution import convolution_2d


class ConvTest(unittest.TestCase):
    def test_simple(self):
        weights = hmarray(
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ])
        ).astype(np.float32)
        a = hmarray(np.random.rand(*(256, 256)).astype(np.float32))
        actual = hmarray(np.zeros(a[1:-1, 1:-1].shape).astype(np.float32))
        convolution_2d(a, weights, actual, (0, 0), (1, 1))
        expected = convolve(a, weights)[1:-1, 1:-1]
        np.testing.assert_allclose(actual, expected)
