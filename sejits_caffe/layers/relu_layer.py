from .base_layer import BaseLayer
from hindemith.utils import symbols
from hindemith.operations.map import hmmap
import numpy as np

class ReLULayer(BaseLayer):
    backend = 'gpu'
    def set_up(self, bottom, top):

      def relu_map(elt):
        if elt > 0:
            return elt
        else:
            return 0

      self.gpu_forward = hmmap(relu_map)

    def cpu_forward(self, bottom, top):
        top[:] = bottom.clip(min=0) + \
            bottom.clip(max=0) * self.layer_param.relu_param.negative_slope

    def forward(self, bottom, top):
        if self.backend == 'gpu':
            self.gpu_forward(bottom, top)
        else:
            self.cpu_forward(bottom, top)

    # def backward(self, top, propagate_down, bottom):
    #     if propagate_down[0]:
    #         data = bottom.data
    #         data[:] = top.diff * (data[data > 0]) \
    #             + self.layer_param.negative_slope * data[data <= 0]
