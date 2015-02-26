import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
import numpy as np
import unittest
from sejits_caffe.layers.relu_layer import ReLULayer
from hindemith import hmarray

path = os.path.dirname(os.path.realpath(__file__))
param = caffe_pb2.NetParameter()
param_string = open(path + '/test.prototxt').read()
text_format.Merge(param_string, param)


@unittest.skip("Need to update")
class ReluLayerTest(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            np.testing.assert_allclose(actual, expected, rtol=1e-04)
        except AssertionError as e:
            self.fail(e)

    def test_forward(self):
        cpu = ReLULayer(param.layers[1])
        cpu.backend = 'cpu'
        gpu = ReLULayer(param.layers[1])
        bottom = np.random.rand(5 * 25 * 256 * 256).astype(np.float32) * 255 - 120
        cpu_top = np.random.rand(5 * 25 * 256 * 256).astype(np.float32) * 255 - 120
        gpu_top = hmarray(np.random.rand(5 * 25 * 256 * 256).astype(np.float32) *
            255)
        gpu.set_up(bottom, gpu_top)

        cpu.forward(bottom, cpu_top)
        gpu.forward(hmarray(bottom), gpu_top)
        gpu_top.copy_to_host_if_dirty()

        self._check(cpu_top, gpu_top)
