from .base_layer_test import LayerTest

from sejits_caffe.layers.conv_layer import ConvLayer
from scipy.ndimage.filters import convolve
import numpy as np
# from hindemith.types.hmarray import hmarray


def numpy_convolve(batch, weights, expected):
    for i in range(batch.shape[0]):
        for n in range(25):
            for channel in range(batch.shape[1]):
                expected[i][n] += convolve(batch[i][channel],
                                           weights)[5:-5, 5:-5]


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
        height_out = (256 - 11) + 1
        width_out = (256 - 11) + 1
        self.actual = np.zeros((5, 25, height_out * width_out), np.float32)
        self.expected = np.zeros((5, 25, height_out, width_out), np.float32)
        conv = ConvLayer(self.in_batch, self.actual, 25, 11)
        conv.forward(self.in_batch, self.actual)
        self.actual = self.actual.reshape((5, 25, height_out, width_out))
        numpy_convolve(self.in_batch, weights, self.expected)
        self._check()
