from sejits_caffe.layers.base_layer import BaseLayer
from sejits_caffe.operations.relu import relu
from sejits_caffe.operations import meta


class ReluLayer(BaseLayer):
    """docstring for ReluLayer"""
    def __init__(self, param):
        super(ReluLayer, self).__init__(param)
        self.negative_slope = param.relu_param.negative_slope

    @meta
    def forward(self, bottom, top):
        for n in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                relu(bottom[n, c], top[n, c], self.negative_slope)
