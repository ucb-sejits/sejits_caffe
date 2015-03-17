from sejits_caffe.layers.base_layer import BaseLayer
from cstructures.array import Array
import numpy as np


class InnerProductLayer(BaseLayer):
    """docstring for InnerProductLayer"""
    def __init__(self, layer_param):
        super(InnerProductLayer, self).__init__(layer_param)
        param = self.layer_param.inner_product_param
        self.num_output = param.num_output
        self.bias_term = param.bias_term
        if self.bias_term:
            self.bias = Array.zeros(self.num_output)
            filler = param.bias_filler
            if filler.type == 'constant':
                self.bias.fill(filler.value)
            else:
                raise Exception("Filler not implemented for bias filler \
                    type {}".format(filler.type))

    def setup(self, bottom, top):
        weights_shape = (self.num_output, bottom.shape[0])
        weight_filler = self.layer_param.inner_product_param.weight_filler
        if weight_filler.type == 'gaussian':
            self.weights = weight_filler.mean + weight_filler.std * \
                Array.standard_normal(
                    weights_shape).astype(np.float32)
        else:
            raise Exception("Filler not implemented for weight filler"
                            "type {}".format(weight_filler.type))

    def forward(self, bottom, top):
        top[:] = np.dot(bottom, self.weights)
        if self.bias_term:
            top += self.bias
