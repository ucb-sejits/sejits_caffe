from sejits_caffe.operations.relu import relu
from cstructures.array import Array
import unittest
import numpy as np


class TestRelu(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_simple(self):
        bottom = Array.rand(256, 256).astype(np.float32) * 255
        actual = Array.zeros(bottom.shape, np.float32)
        relu(bottom, bottom, actual, 0.0)
        expected = np.clip(bottom, 0.0, float('inf'))
        self._check(actual, expected)

    def test_nonzero_slope(self):
        bottom = Array.rand(256, 256).astype(np.float32) * 255
        actual = Array.zeros(bottom.shape, np.float32)
        relu(bottom, bottom, actual, 2.4)
        expected = np.clip(bottom, 0.0, float('inf')) + \
            2.4 * np.clip(bottom, float('-inf'), 0.0)
        self._check(actual, expected)

if __name__ == '__main__':
    unittest.main()
