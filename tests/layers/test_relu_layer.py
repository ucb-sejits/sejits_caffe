import unittest
from sejits_caffe.types import Array
from sejits_caffe.layers.relu_layer import ReluLayer
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
import numpy as np


path = os.path.dirname(os.path.realpath(__file__))


class TestReluLayer(unittest.TestCase):
    def setUp(self):
        param_string = open(path + '/alexnet.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        self.layer = param.layer

    def test_forward_simple(self):
        channels = 12
        height = 3
        width = 5
        bottom = Array.rand(
            5, channels, height, width).astype(np.float32)
        bottom = bottom * 256 - 128
        layer = ReluLayer(self.layer[3])
        actual = Array.zeros(bottom.shape, np.float32)
        layer.forward(bottom, actual)
        expected = np.clip(bottom, 0.0, float('inf')).astype(np.float32)
        np.testing.assert_allclose(actual, expected)

    # def test_backward_simple(self):
    #     channels = 12
    #     height = 3
    #     width = 5
    #     bottom = Array.rand(
    #         5, channels, height, width).astype(np.float32)
    #     bottom = bottom * 256 - 128
    #     layer = ReluLayer(self.layer[3])
    #     actual = Array.zeros(bottom.shape, np.float32)
    #     layer.backward(bottom, actual, top, top_diff)
    #     expected = np.multiply(top_diff, np.greater(bottom, np.zeros(bottom)))
    #     np.testing.assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()
