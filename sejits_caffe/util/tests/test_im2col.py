import unittest
from sejits_caffe.util.im2col import cpu_im2col, gpu_im2col
from hindemith.types.hmarray import hmarray
import numpy as np


def py_im2col(a, kernel_size, pad, stride):
    h_out = (a.shape[1] + 2 * pad - kernel_size) // stride + 1
    w_out = (a.shape[2] + 2 * pad - kernel_size) // stride + 1
    out_shape = (3 * kernel_size * kernel_size,
                 h_out * w_out)
    out = np.ndarray(out_shape, np.float32)
    height_col = (a.shape[1] + 2 * pad - kernel_size) // stride + 1
    channels_col = kernel_size * kernel_size * a.shape[0]
    for c in range(channels_col):
        w_offset = c % kernel_size
        h_offset = (c / kernel_size) % kernel_size
        c_im = c / kernel_size / kernel_size
        for h in range(0, height_col):
            for w in range(0, height_col):
                h_pad = h * stride - pad + h_offset
                w_pad = w * stride - pad + w_offset
                if h_pad >= 0 and h_pad < a.shape[1] and \
                   w_pad >= 0 and w_pad < a.shape[2]:
                    out[c, h * height_col + w] = a[c_im, h_pad, w_pad]
                else:
                    out[c, h * height_col + w] = 0
    return out


class TestIm2Col(unittest.TestCase):
    def setUp(self):
        height = 64
        width = 64
        self.a = hmarray(
            np.random.rand(3, height, width) * 100).astype(np.float32)
        # self.a = hmarray(
        #     (np.arange(3 * height * width).reshape(
        #         3, height, width) * 100).astype(np.float32))
        self.kernel_size = 11
        self.pad = 0
        self.stride = 1

    def test_c(self):
        actual = cpu_im2col(self.a, self.a.shape, (11, 11),
                            (0, 0), (1, 1))
        expected = py_im2col(self.a, 11, 0, 1)
        np.testing.assert_allclose(actual, expected)

    def test_ocl(self):
        actual = gpu_im2col(self.a, self.a.shape, (11, 11),
                            (0, 0), (1, 1))
        expected = py_im2col(self.a, 11, 0, 1)
        actual.copy_to_host_if_dirty()
        np.testing.assert_allclose(actual, expected)
