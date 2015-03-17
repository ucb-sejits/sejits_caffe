import unittest
from cstructures.array import Array
from sejits_caffe.layers.lrn_layer import LRNLayer
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format

import numpy as np
import os


path = os.path.dirname(os.path.realpath(__file__))


class TestLRNLayer(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            np.testing.assert_allclose(actual, expected, atol=1e-03)
        except AssertionError as e:
            self.fail(e)

    def setUp(self):
        param_string = open(path + '/alexnet.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        self.layer = param.layer

    def test_simple(self):
        bottom = Array.rand(3, 8, 32, 32).astype(np.float32)
        actual = Array.zeros_like(bottom)
        layer = LRNLayer(self.layer[4])
        param = layer.layer_param.lrn_param
        alpha = param.alpha
        size = param.local_size
        beta = param.beta
        layer.setup(bottom, actual)
        layer.forward(bottom, actual)
        expected = Array.zeros_like(bottom)
        for n in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                for h in range(bottom.shape[2]):
                    for w in range(bottom.shape[3]):
                        c_start = c - (size - 1) // 2
                        c_end = min(c_start + size, bottom.shape[1])
                        scale = 1
                        for i in range(c_start, c_end):
                            value = bottom[n, i, h, w]
                            scale += value * value * alpha / size
                        expected = bottom[n, c, h, w] / pow(scale, beta)
                        self.assertTrue(
                            abs(actual[n, c, h, w] - expected) < 1e-4)


if __name__ == '__main__':
    unittest.main()
