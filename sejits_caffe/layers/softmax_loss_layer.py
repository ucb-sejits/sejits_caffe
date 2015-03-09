from sejits_caffe.layers.loss_layer import LossLayer
from sejits_caffe.layers.base_layer import BaseLayer
from sejits_caffe.types import Array
import numpy as np


class SoftMaxLayer(BaseLayer):
    """docstring for SoftMaxLayer"""
    def __init__(self, param):
        super(SoftMaxLayer, self).__init__(param)
        self.param = param

    def setup(self, bottom, top):
        self.sum_multiplier = Array.ones((bottom.shape[1]), np.float32)

    def forward(self, bottom, top):
        top[:] = bottom[:]
        spatial_dim = bottom.shape[2] * bottom.shape[3]
        for n in range(bottom.shape[0]):
            scale = bottom[n][0]

            # Initialize scale to the first plane
            for c in range(1, bottom.shape[1]):
                scale = np.maximum(scale, bottom[n, c])

            for c in range(bottom.shape[1]):
                top[n, c] -= self.sum_multiplier[c] + scale
            top[n] = np.exp(top[n])

            for h in range(bottom.shape[2]):
                for w in range(bottom.shape[3]):
                    scale[h, w] = np.dot(top[n].T[w, h], self.sum_multiplier)

            for j in range(bottom.shape[1]):
                top[n, j] /= scale



class SoftMaxWithLossLayer(LossLayer):
    """docstring for SoftMaxWithLossLayer"""
    def __init__(self, param):
        super(SoftMaxWithLossLayer, self).__init__(param)




