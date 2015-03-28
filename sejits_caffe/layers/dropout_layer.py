from sejits_caffe.layers.base_layer import BaseLayer
import numpy as np


class DropoutLayer(BaseLayer):
    """docstring for DropoutLayer"""
    def __init__(self, param):
        super(DropoutLayer, self).__init__(param)
        self.threshold = param.dropout_param.dropout_ratio
        assert 0 <= self.threshold <= 1, "Threshold must be between 0 and 1"
        self.scale = 1.0 / (1.0 - self.threshold)
        self.propagate_down = True
        self.mask = None
        self.phase = 'train'

    def get_top_shape(self, bottom):
        return bottom.shape

    def forward(self, bottom, top):
        self.mask = np.random.binomial(bottom.shape[0], 1.0 - self.threshold,
                                       bottom.shape)
        top[:] = bottom * self.mask * self.scale

    def backward(self, bottom_diff, top_diff):
        if self.propagate_down:
            if self.phase == 'train':
                bottom_diff[:] = top_diff * self.mask * self.scale
            else:
                bottom_diff[:] = top_diff
