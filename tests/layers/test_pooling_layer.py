import unittest
from sejits_caffe.types import Array
from sejits_caffe.layers.pooling_layer import PoolingLayer
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
import numpy as np


path = os.path.dirname(os.path.realpath(__file__))


class TestPoolingLayer(unittest.TestCase):
    def setUp(self):
        param_string = open(path + '/test_pool.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        self.layers = param.layers

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
                     [1, 2, 5, 2, 3]], np.int32)
        layer = PoolingLayer(self.layers[0])
        pooled_height = (height + 2 * layer.pad_h - layer.kernel_h) \
            / layer.stride_h + 1
        pooled_width = (width + 2 * layer.pad_w - layer.kernel_w) \
            / layer.stride_w + 1
        actual = Array.zeros((5, channels, pooled_height, pooled_width),
                             np.int32)
        layer.setup(bottom, actual)
        layer.forward(bottom, actual)
        for n in range(5):
            for c in range(channels):
                np.testing.assert_array_equal(
                    actual[n, c],
                    np.array([
                        [9, 5, 5, 8],
                        [9, 5, 5, 8]
                    ], np.int32))


if __name__ == '__main__':
    unittest.main()