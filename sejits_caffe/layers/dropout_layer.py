from sejits_caffe.layers.base_layer import BaseLayer
import numpy as np


class DropoutLayer(BaseLayer):
    """docstring for DropoutLayer"""
    def __init__(self, param):
        super(DropoutLayer, self).__init__(param)
        self.threshold = param.dropout_param.dropout_ratio
        assert 0 <= self.threshold <= 1, "Threshold must be between 0 and 1"
        self.scale = 1.0 / (1.0 - self.threshold)

    def forward(self, bottom, top):
        mask = np.random.binomial(bottom.shape[0], 1.0 - self.threshold,
                                  bottom.shape)
        top[:] = bottom * mask * self.scale
