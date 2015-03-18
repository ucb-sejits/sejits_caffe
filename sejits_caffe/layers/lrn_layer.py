from sejits_caffe.layers.base_layer import BaseLayer
from cstructures.array import Array
import numpy as np


class LRNLayer(BaseLayer):
    """docstring for LRNLayer"""
    def __init__(self, param):
        super(LRNLayer, self).__init__(param)
        param = param.lrn_param
        self.size = param.local_size
        assert self.size % 2 == 1, \
            "LRN only supports odd values for local size"

        self.pre_pad = (self.size - 1) / 2
        self.alpha = param.alpha
        self.beta = param.beta
        self.k = param.k

    def setup(self, bottom, top):
        self.scale = Array.zeros_like(bottom)

    def forward(self, bottom, top):
        # initialize scale to constant value
        self.scale.fill(self.k)

        padded_square = Array.zeros((bottom.shape[1] + self.size - 1,
                                     bottom.shape[2], bottom.shape[3]),
                                    bottom.dtype)

        alpha_over_size = self.alpha / self.size

        for n in range(bottom.shape[0]):
            padded_square[self.pre_pad:bottom.shape[1]+self.pre_pad] = \
                np.square(bottom[n])

            for c in range(self.size):
                self.scale[n] += alpha_over_size * padded_square[c]

            for c in range(1, bottom.shape[1]):
                self.scale[n, c] = self.scale[n, c - 1]

                self.scale[n, c] += \
                    alpha_over_size * padded_square[c + self.size - 1]

                self.scale[n, c] -= alpha_over_size * padded_square[c - 1]

        top[:] = np.power(self.scale, -self.beta) * bottom

    def backward(self, bottom_data, bottom_diff, top_data, top_diff):
        padded_ratio = Array.zeros(1, bottom_data.shape[1] + self.size - 1,
                                   bottom_data.shape[2], bottom_data.shape[3])
        accum_ratio = Array.zeros(1, 1, bottom_data.shape[2],
                                  bottom_data.shape[3])
        accum_ratio_times_bottom = Array.zeros(1, 1, bottom_data.shape[2],
                                               bottom_data.shape[3])
        cache_ratio_value = 2.0 * self.apha * self.beta / self.size
        bottom_diff = np.pow(self.scale, -self.beta)
        bottom_diff *= top_diff

        inverse_pre_pad = self.size - (self.size + 1) / 2
        for n in range(bottom_data.shape[0]):
            padded_ratio[0, inverse_pre_pad] = top_diff[n] * top_data[n]
            padded_ratio[0, inverse_pre_pad] /= self.scale[n]
            accum_ratio.fill(0)
            for c in range(self.size - 1):
                accum_ratio += padded_ratio[0, c]

            for c in range(bottom_data.shape[1]):
                accum_ratio += padded_ratio[0, c + self.size - 1]
                accum_ratio_times_bottom += bottom_data[n, c] * accum_ratio
                bottom_data[n, c] += -cache_ratio_value * \
                    accum_ratio_times_bottom
                accum_ratio += -1 * padded_ratio[0, c]
