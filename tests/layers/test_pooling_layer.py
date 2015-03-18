import unittest
from cstructures.array import Array
from sejits_caffe.layers.pooling_layer import PoolingLayer
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
import numpy as np


path = os.path.dirname(os.path.realpath(__file__))


class TestPoolingLayer(unittest.TestCase):
    def setUp(self):
        param_string = open(path + '/alexnet.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        self.layer = param.layer

    def test_simple(self):
        channels = 12
        height = 3
        width = 5
        bottom = Array.zeros((5, channels, height, width), np.int32)
        for n in range(5):
            for c in range(channels):
                bottom[n, c] = Array.array(
                    [[1, 2, 5, 2, 3],
                     [9, 4, 1, 4, 8],
                     [1, 2, 5, 2, 3]]).astype(np.int32)
        param = self.layer[5]
        param.pooling_param.kernel_size = 2
        param.pooling_param.stride = 1
        layer = PoolingLayer(param)
        actual_shape = layer.get_top_shape(bottom)
        actual = Array.zeros(actual_shape, np.int32)
        layer.setup(bottom, actual)
        layer.forward(bottom, actual)
        for n in range(5):
            for c in range(channels):
                np.testing.assert_array_equal(
                    actual[n, c],
                    np.array([
                        [9, 5, 5, 8],
                        [9, 5, 5, 8]
                    ]).astype(np.int32))
        bottom = Array.zeros_like(bottom)
        for n in range(5):
            for c in range(channels):
                actual[n, c] = Array.array(
                    [[1, 1, 1, 1],
                     [1, 1, 1, 1]]).astype(np.int32)
        layer.backward(bottom, actual)
        for n in range(5):
            for c in range(channels):
                np.testing.assert_array_equal(
                    bottom[n, c],
                    np.array([[0, 0, 2, 0, 0],
                              [2, 0, 0, 0, 2],
                              [0, 0, 2, 0, 0]]).astype(np.int32))

if __name__ == '__main__':
    unittest.main()
