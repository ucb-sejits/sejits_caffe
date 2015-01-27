from .base_layer_test import LayerTest

from sejits_caffe.layers.conv_layer import ConvLayer
from scipy.ndimage.filters import convolve
import numpy as np
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
# from hindemith.types.hmarray import hmarray


def numpy_convolve(batch, weights, expected):
    for i in range(batch.shape[0]):
        for n in range(25):
            for channel in range(batch.shape[1]):
                expected[i][n] += convolve(batch[i][channel],
                                           weights[n][channel])[5:-5, 5:-5]


class ConvLayerTest(LayerTest):
    def test_forward(self):
        height_out = (256 - 11) + 1
        width_out = (256 - 11) + 1
        self.actual = np.zeros((5, 25, height_out * width_out), np.float32)
        self.expected = np.zeros((5, 25, height_out, width_out), np.float32)

        path = os.path.dirname(os.path.realpath(__file__))
        param = caffe_pb2.NetParameter()
        param_string = open(path + '/test.prototxt').read()
        text_format.Merge(param_string, param)
        conv = ConvLayer(param.layers[0])
        conv.set_up(self.in_batch, self.actual)
        conv.forward(self.in_batch, self.actual)
        self.actual = self.actual.reshape((5, 25, height_out, width_out))
        new_weights = conv.blobs[0].data.reshape(25, 3, 11, 11)
        numpy_convolve(self.in_batch, new_weights, self.expected)
        self._check()
