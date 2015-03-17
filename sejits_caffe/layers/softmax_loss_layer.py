from sejits_caffe.layers.loss_layer import LossLayer
from sejits_caffe.layers.base_layer import BaseLayer
from cstructures.array import Array
import sejits_caffe.caffe_pb2 as caffe_pb2
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
        for n in range(bottom.shape[0]):
            # Initialize scale to the first plane
            scale = bottom[n, 0]

            for c in range(1, bottom.shape[1]):
                scale = np.maximum(scale, bottom[n, c])

            for c in range(bottom.shape[1]):
                top[n, c] -= self.sum_multiplier[c] * scale

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
        softmax_param = caffe_pb2.SoftmaxParameter()
        self.softmax_layer = SoftMaxLayer(softmax_param)

        self.ignore_label = param.loss_param.ignore_label
        self.normalize = param.loss_param.normalize

    def setup(self, bottom_data, bottom_label, top):
        self.prob = Array.zeros_like(bottom_data)
        self.softmax_layer.setup(bottom_data, self.prob)

    def forward(self, bottom_data, bottom_label, top):
        self.softmax_layer.forward(bottom_data, self.prob)
        label = bottom_label
        loss = 0.0
        count = 0
        for i in range(self.prob.shape[0]):
            for j in range(self.prob.shape[2]):
                for k in range(self.prob.shape[3]):
                    label_val = label[i, 0, j, k]
                    if self.ignore_label == label_val:
                        continue
                    assert 0 <= label_val < self.prob.shape[1]
                    loss -= np.log(max(self.prob[i, label_val, j, k],
                                       np.finfo(np.float32).eps))
                    count += 1

        if self.normalize:
            top[0] = loss / count
        else:
            top[0] = loss / bottom_data.shape[0]
