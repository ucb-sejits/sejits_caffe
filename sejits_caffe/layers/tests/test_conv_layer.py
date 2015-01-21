from .base_layer_test import LayerTest

from sejits_caffe.layers.conv_layer import ConvLayer
from scipy.ndimage.filters import convolve
import numpy as np
from hindemith.types.hmarray import hmarray


def numpy_convolve(batch, weights, expected):
    for i in range(batch.shape[0]):
        for channel in range(batch.shape[1]):
            expected[i][channel] = convolve(batch[i][channel], weights)


class ConvLayerTest(LayerTest):
    def test_simple(self):
        weights = np.array([
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
        ], np.float32)
        conv = ConvLayer([weights], self.in_batch, self.actual, 96, 11)
        self.actual = hmarray(np.zeros((5, 96, conv.height_out, conv.width_out), np.float32))
        self.expected = hmarray(np.zeros((5, 96, conv.height_out, conv.width_out), np.float32))
        conv.forward(self.in_batch, self.actual)
        numpy_convolve(self.in_batch, weights, self.expected)
        self._check()
