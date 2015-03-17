import unittest
from cstructures.array import Array
from sejits_caffe.layers.softmax_loss_layer import SoftMaxLayer, \
    SoftMaxWithLossLayer
import numpy as np
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os

path = os.path.dirname(os.path.realpath(__file__))


class TestSoftmaxLayer(unittest.TestCase):
    def test_simple(self):
        bottom = Array.rand(2, 10, 2, 3).astype(np.float32)
        top = Array.zeros_like(bottom)
        layer = SoftMaxLayer(None)
        layer.setup(bottom, top)
        layer.forward(bottom, top)
        for i in range(bottom.shape[0]):
            for k in range(bottom.shape[2]):
                for l in range(bottom.shape[3]):
                    _sum = np.sum(top[i, ..., k, l])
                    self.assertTrue(_sum > 0.999)
                    self.assertTrue(_sum < 1.001)

                    scale = np.sum(np.exp(bottom[i, ..., k, l]))
                    for j in range(bottom.shape[1]):
                       self.assertTrue(top[i, j, k, l] + 1e-4 >=
                                       np.exp(bottom[i, j, k, l]) / scale)
                       self.assertTrue(top[i, j, k, l] - 1e-4 <=
                                       np.exp(bottom[i, j, k, l]) / scale)


@unittest.skip("Not working")
class TestSoftmaxWitLossLayer(unittest.TestCase):
    def test_forward(self):
        param_string = open(path + '/alexnet.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        param = param.layer[-1]
        param.loss_param.normalize = False
        layer = SoftMaxWithLossLayer(param)
        bottom = [Array.rand(10, 5, 2, 3).astype(np.float32) * 10,
                  (Array.rand(10, 1, 2, 3) * 5).astype(np.int32)]
        top = Array.zeros((1, ), np.float32)
        layer.setup(bottom, top)
        layer.forward(bottom, top)
        full_loss = top[0]
        accum_loss = 0.0
        for label in range(5):
            param.loss_param.ignore_label = label
            layer = SoftMaxWithLossLayer(param)
            layer.setup(bottom, top)
            layer.forward(bottom, top)
            print(top[0])
            accum_loss += top[0]

        print(full_loss)
        print(accum_loss)
        self.assertTrue(abs(4 * full_loss - accum_loss) < 1e-4)



if __name__ == '__main__':
    unittest.main()
