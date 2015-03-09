import unittest
from sejits_caffe.types import Array
from sejits_caffe.layers.softmax_loss_layer import SoftMaxLayer
import numpy as np


class TestSoftmaxLayer(unittest.TestCase):
    def test_simple(self):
        bottom = Array.rand(2, 10, 2, 3).astype(np.float32)
        top = Array.zeros_like(bottom)
        layer = SoftMaxLayer(None)
        layer.setup(bottom, top)
        layer.forward(bottom, top)
        for i in range(bottom.shape[0]):
            for k in range(bottom.shape[2]):
                for l in range(bottom.shape[3]):
                    _sum = np.sum(top[i, ..., k, l])
                    self.assertTrue(_sum > 0.999)
                    self.assertTrue(_sum < 1.001)

                    scale = np.sum(np.exp(bottom[i, ..., k, l]))
                    for j in range(bottom.shape[1]):
                       self.assertTrue(top[i, j, k, l] + 1e-4 >=
                                       np.exp(bottom[i, j, k, l]) / scale)
                       self.assertTrue(top[i, j, k, l] - 1e-4 <=
                                       np.exp(bottom[i, j, k, l]) / scale)


if __name__ == '__main__':
    unittest.main()
