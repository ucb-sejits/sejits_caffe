from sejits_caffe.layers.conv_layer import ConvLayer
from sejits_caffe.types import Array
import numpy as np
import sejits_caffe.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import os
import unittest
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import CFile, Constant
from ctree.nodes import Project
from ctree.templates.nodes import FileTemplate
import ctypes as ct
import ast


class ConvConcrete(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        from ctree.util import Timer
        with Timer() as t:
            self._c_function(*args)
        print("Naive time {}".format(t.interval))


class NaiveConv(LazySpecializedFunction):
    def __init__(self, conv_param):
        super(NaiveConv, self).__init__(ast.Module())
        self.conv_param = conv_param

    def args_to_subconfig(self, args):
        cfg = {}
        for name, arg in zip(['in_ptr', 'weights', 'bias', 'out'], args):
            cfg[name] = np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)
        return cfg

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        conv_param = self.conv_param
        kernel_size = conv_param.kernel_size
        pad = conv_param.pad
        stride = conv_param.stride
        group = conv_param.group
        out_num, out_c, out_h, out_w = arg_cfg['out']._shape_
        in_ptr_num, in_ptr_c, in_ptr_h, in_ptr_w = arg_cfg['in_ptr']._shape_
        weights_g, weights_c, weights_h, weights_w = arg_cfg['weights']._shape_
        return [CFile('conv', [
            FileTemplate(
                os.path.dirname(os.path.realpath(__file__)) +
                '/conv_test.tmpl.c',
                {
                    'kernel_size': Constant(kernel_size),
                    'pad': Constant(pad),
                    'stride': Constant(stride),
                    'group': Constant(group),
                    'out_num': Constant(out_num),
                    'out_c': Constant(out_c),
                    'out_h': Constant(out_h),
                    'out_w': Constant(out_w),
                    'in_num': Constant(in_ptr_num),
                    'in_c': Constant(in_ptr_c),
                    'in_h': Constant(in_ptr_h),
                    'in_w': Constant(in_ptr_w),
                    'weight_g': Constant(weights_g),
                    'weight_c': Constant(weights_c),
                    'weight_h': Constant(weights_h),
                    'weight_w': Constant(weights_w),
                    'bias_term': Constant(1 if conv_param.bias_term else 0)
                }
            )], config_target='omp')]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        entry_type = (None, )
        for name in ['in_ptr', 'weights', 'bias', 'out']:
            entry_type += (arg_cfg[name], )
        fn = ConvConcrete('conv', proj, ct.CFUNCTYPE(*entry_type))
        return fn


path = os.path.dirname(os.path.realpath(__file__))


class ConvLayerTest(unittest.TestCase):
    def _check(self, actual, expected):
        try:
            np.testing.assert_allclose(actual, expected, atol=1e-03)
        except AssertionError as e:
            self.fail(e)

    def setUp(self):
        param_string = open(path + '/alexnet.prototxt').read()
        param = caffe_pb2.NetParameter()
        text_format.Merge(param_string, param)
        self.layer = param.layer

    def _forward_test(self, param, in_shape):
        conv_param = param.convolution_param
        num_output = conv_param.num_output
        kernel_size = conv_param.kernel_size
        height_out = (in_shape[2] + 2 * conv_param.pad - kernel_size) // \
            conv_param.stride + 1
        width_out = (in_shape[3] + 2 * conv_param.pad - kernel_size) // \
            conv_param.stride + 1
        actual_shape = (in_shape[0], num_output, height_out, width_out)
        expected_shape = (in_shape[0], num_output, height_out, width_out)
        conv = ConvLayer(param)
        expected_conv = NaiveConv(conv_param)
        actual = Array.zeros(actual_shape, np.float32)
        expected = Array.zeros(expected_shape, np.float32)
        in_batch = Array.rand(*in_shape).astype(np.float32) * 255

        conv.set_up(in_batch, actual)
        conv.forward(in_batch, actual)
        expected_conv(in_batch, conv.weights, conv.bias, expected)
        self._check(actual, expected)

    def test_alex_net_conv1(self):
        self._forward_test(self.layer[2], (5, 3, 256, 256))

    def test_alex_net_conv2(self):
        self._forward_test(self.layer[6], (5, 16, 64, 64))

    def test_alex_net_conv3(self):
        self._forward_test(self.layer[10], (5, 4, 64, 64))

    def test_alex_net_conv4(self):
        self._forward_test(self.layer[12], (5, 8, 64, 64))

    def test_alex_net_conv5(self):
        self._forward_test(self.layer[14], (5, 8, 64, 64))

if __name__ == '__main__':
    unittest.main()
