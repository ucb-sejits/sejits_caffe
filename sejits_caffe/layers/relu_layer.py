from base_layer import BaseLayer
import numpy as np


class ReLULayer(BaseLayer):
    def forward(self, bottom, top):
        top.data[:] = np.maximum(bottom.data, 0) + \
            self.layer_param.negative_slope * np.minimum(bottom.data, 0)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            data = bottom.data
            data[:] = top.diff * (data[data > 0]) \
                + self.layer_param.negative_slope * data[data <= 0]
