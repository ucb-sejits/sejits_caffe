from sejits_caffe.layers.base_layer import BaseLayer
from sejits_caffe.types import Array
import numpy as np


class AccuracyLayer(BaseLayer):
    """docstring for AccuracyLayer"""
    def __init__(self, param):
        super(AccuracyLayer, self).__init__(param)
        self.top_k = param.accuracy_param.top_k

    def forward(self, bottom, top):
        accuracy = 0
        data = bottom[0]
        label = bottom[1]
        dim = np.prod(data.shape) / data.shape[0]
        for i in range(data.shape[0]):
            vec = Array.array(
                [[data[i * dim + j], j] for j in range(dim)])
            vec.partition((0, self.top_k))

        for k in range(self.top_k):
            if vec[k][1] == label[i]:
                accuracy += 1

        top[0] = accuracy / data.shape[0]
