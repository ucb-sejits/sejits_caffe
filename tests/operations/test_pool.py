from sejits_caffe.operations import max_pool
from cstructures.array import Array
import unittest
import numpy as np


def py_max_pool(data, output, mask, kernel_size, stride, pad):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = pad
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            y_start = max(y * stride_h - pad_h, 0)
            x_start = max(x * stride_w - pad_w, 0)
            y_end = min(y_start + kernel_h, data.shape[0])
            x_end = min(x_start + kernel_w, data.shape[1])
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    if data[yy, xx] > output[y, x]:
                        output[y, x] = data[yy, xx]
                        mask[y, x] = yy * data.shape[1] + xx


class TestPooling(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        a = Array.rand(256, 256).astype(np.float32) * 255
        actual_mask = Array.zeros((254, 254), np.float32)
        actual = Array.zeros((254, 254), np.float32)
        actual.fill(float('-inf'))
        expected_mask = Array.zeros((254, 254), np.float32)
        expected = Array.zeros((254, 254), np.float32)
        expected.fill(float('-inf'))

        max_pool(a, actual, actual_mask, (2, 2))
        py_max_pool(a, expected, expected_mask,
                    (2, 2), (1, 1), (0, 0))
        self._check(actual, expected)
        self._check(actual_mask, expected_mask)

if __name__ == '__main__':
    unittest.main()
