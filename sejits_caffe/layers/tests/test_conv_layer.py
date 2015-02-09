from .base_layer_test import LayerTest

from sejits_caffe.layers.conv_layer import ConvLayer
from scipy.ndimage.filters import convolve
import numpy as np
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
from hindemith.types.hmarray import hmarray


def numpy_convolve(batch, weights, expected):
    for i in range(batch.shape[0]):
        for n in range(25):
            for channel in range(batch.shape[1]):
                expected[i][n] += convolve(batch[i][channel],
                                           weights[n][channel])[5:-5, 5:-5]


path = os.path.dirname(os.path.realpath(__file__))
param = caffe_pb2.NetParameter()
param_string = open(path + '/test.prototxt').read()
text_format.Merge(param_string, param)
conv_param = param.layers[0].convolution_param
height_out = (256 - conv_param.kernel_size) + 1
width_out = (256 - conv_param.kernel_size) + 1
actual_shape = (5, conv_param.num_output, height_out * width_out)
expected_shape = (5, conv_param.num_output, height_out, width_out)


class ConvLayerTest(LayerTest):
    def _check(self, actual, expected):
        try:
            np.testing.assert_allclose(actual, expected, rtol=1e-04)
        except AssertionError as e:
            self.fail(e)

    def _forward_test(self, backend):
        conv = ConvLayer(param.layers[0])
        conv.backend = backend
        actual = hmarray(np.zeros(actual_shape, np.float32))
        expected = np.zeros(expected_shape, np.float32)
        in_batch = np.random.rand(5, 3, 256, 256).astype(np.float32) * 255

        conv.set_up(hmarray(in_batch), actual)
        conv.forward(hmarray(in_batch), actual)
        actual = actual.reshape((5, 25, height_out, width_out))
        new_weights = conv.weights.reshape(25, 3, 11, 11)
        numpy_convolve(in_batch, new_weights, expected)
        self._check(actual, expected)

    def test_cpu_forward(self):
        self._forward_test('cpu')

    def test_gpu_forward(self):
        self._forward_test('gpu')
