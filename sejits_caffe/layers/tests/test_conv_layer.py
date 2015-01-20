from .base_layer_test import LayerTest

from sejits_caffe.layers.conv_layer import ConvLayer
from scipy.ndimage.filters import convolve


def numpy_convolve(batch, weights, expected):
    for i in range(batch.shape[0]):
        expected[i] = convolve(batch[i], weights)


class ConvLayerTest(LayerTest):
    def test_simple(self):
        conv = ConvLayer(self.in_batch, self.actual)
        conv.forward(self.in_batch, self.actual)
        numpy_convolve(self.in_batch, conv.weights, self.expected)
        self._check()
