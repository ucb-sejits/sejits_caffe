import unittest
from sejits_caffe.types import Array
from sejits_caffe.operations.convolution import convolve
from scipy import signal

import numpy as np


class TestConvolution(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_simple(self):
        a = (Array.rand(256, 256) * 255).astype(np.float32)
        weights = (Array.rand(3, 3) * 2).astype(np.float32)
        actual = Array.zeros(a.shape, np.float32)
        convolve(a, weights, actual, (1, 1), (1, 1))

        expected = signal.convolve(a, np.fliplr(np.flipud(weights)),
                                   mode='same')

        self._check(actual, expected)

    def test_no_padding(self):
        a = (Array.rand(256, 256) * 255).astype(np.float32)
        weights = (Array.rand(5, 5) * 2).astype(np.float32)
        actual = Array.zeros((254, 254), np.float32)
        convolve(a, weights, actual, (0, 0), (1, 1))

        expected = signal.convolve(a, np.fliplr(np.flipud(weights)),
                                   mode='same')[2:, 2:]
        self._check(actual, expected)


if __name__ == '__main__':
    unittest.main()
