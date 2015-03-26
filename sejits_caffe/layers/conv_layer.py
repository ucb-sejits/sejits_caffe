from .base_layer import BaseLayer
import numpy as np
import logging
# from sejits_caffe.util.im2col import cpu_im2col, gpu_im2col
# from sejits_caffe.operations import convolve, meta
from cstructures.array import Array  # , specialize
from sejits_caffe.util.im2col import Im2Col

# from hindemith.operations.gemm import gemm
# import ctypes
# from ctypes import c_int, c_float, c_size_t

"""
import pycl as cl
import os
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    ext = "so"
elif _platform == "darwin":
    ext = "dylib"

_blaslib = ctypes.cdll.LoadLibrary("libcblas.{}".format(ext))
path = os.path.dirname(os.path.abspath(__file__))
_clblaslib = ctypes.cdll.LoadLibrary(path + "/libclBLAS.{}".format(ext))


def cpu_gemm(A, A_offset, B, B_offset, C, C_offset, m, n, k):

    cblas_row_major = c_int(101)
    no_trans = c_int(111)
    m = c_int(int(m))
    n = c_int(int(n))
    k = c_int(int(k))
    one = c_float(1.0)
    zero = c_float(0.0)
    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    A_ptr.value += A_offset * A.itemsize
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    B_ptr.value += B_offset * B.itemsize
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)
    C_ptr.value += C_offset * C.itemsize

    _blaslib.cblas_sgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                         one, A_ptr, k, B_ptr, n, zero, C_ptr, n)


from ctree.ocl import get_context_and_queue_from_devices
devices = cl.clGetDeviceIDs()
context, queue = get_context_and_queue_from_devices([devices[-1]])
err = _clblaslib.clblasSetup()


def gpu_gemm(A, A_offset, B, B_offset, C, C_offset, m, n, k):
    cblas_row_major = c_int(0)
    no_trans = c_int(0)
    m = c_size_t(int(m))
    n = c_size_t(int(n))
    k = c_size_t(int(k))
    one = c_float(1.0)
    zero = c_float(0.0)

    _clblaslib.clblasSgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                           one, A.ocl_buf, c_size_t(A_offset), k, B.ocl_buf,
                           c_size_t(B_offset), n, zero, C.ocl_buf,
                           c_size_t(C_offset), n,
                           c_size_t(1), ctypes.byref(queue), c_size_t(0),
                           None, None)
    C._host_dirty = True
    cl.clFinish(queue)
"""


class ConvLayer(BaseLayer):
    def __init__(self, param):
        super(ConvLayer, self).__init__(param)

        conv_param = param.convolution_param

        if conv_param.kernel_size:
            self.kernel_h = conv_param.kernel_size
            self.kernel_w = conv_param.kernel_size
        else:
            self.kernel_h = conv_param.kernel_h
            self.kernel_w = conv_param.kernel_w
        assert (self.kernel_h, self.kernel_w) > (0, 0), \
            "Filter dimensions cannot be zero."

        self.padding = (conv_param.pad, conv_param.pad)
        self.stride = (conv_param.stride, conv_param.stride)

        assert conv_param.num_output > 0, "Layer must have at least one output"

        self.group = conv_param.group

        self.weights = None
        self.bias_term = None
        self.bias = None
        self.kernel_size = self.kernel_h, self.kernel_w
        self.im2col = Im2Col(self.kernel_size, self.stride, self.padding)

    def get_top_shape(self, bottom):
        conv_param = self.layer_param.convolution_param
        height_out = (bottom.shape[2] + 2 * self.padding[0] - self.kernel_h) // \
            self.stride[0] + 1
        width_out = (bottom.shape[3] + 2 * self.padding[1] - self.kernel_w) // \
            self.stride[1] + 1
        return bottom.shape[0], conv_param.num_output, height_out, width_out

    def set_up(self, bottom, top):
        conv_param = self.layer_param.convolution_param

        channels, height, width = bottom[0].shape
        num_output = conv_param.num_output
        assert channels % self.group == 0, \
            "Number of channels should be a multiple of group."
        assert num_output % self.group == 0, \
            "Number of outputs should be a multiple of group."

        self.bias_term = conv_param.bias_term
        if self.weights is not None:
            logging.debug("Skipping parameter initialization")
        else:
            weights_shape = (num_output, channels // self.group,
                             self.kernel_h, self.kernel_w)
            weight_filler = conv_param.weight_filler
            if weight_filler.type == 'gaussian':
                self.weights = weight_filler.mean + weight_filler.std * \
                    Array.standard_normal(
                        weights_shape).astype(np.float32)
                self.weight_diff = Array.empty_like(self.weights)
            else:
                raise Exception("Filler not implemented for weight filler \
                    type {}".format(weight_filler.type))
            if self.bias_term:
                self.bias = Array((num_output, ), np.float32)
                self.bias_diff = Array.empty_like(self.bias)
                filler = conv_param.bias_filler
                if filler.type == 'constant':
                    self.bias.fill(filler.value)
                else:
                    raise Exception("Filler not implemented for bias filler \
                        type {}".format(filler.type))

    # @meta
    def forward(self, bottom, top):
        weights = self.weights.reshape(self.weights.shape[0],
                                       np.prod(self.weights.shape[1:]))
        for bottom_data, top_data in zip(bottom, top):
            self.col_data = self.im2col(
                bottom_data, self.kernel_size, self.padding, self.stride)
            col_offset = self.col_data.shape[0] // self.group
            weight_offset = top_data.shape[0] // self.group
            top_offset = top_data.shape[0] // self.group
            for g in range(self.group):
                top_data[g*top_offset:(g+1)*top_offset] = (
                    weights[g*weight_offset:(g+1)*weight_offset].dot(
                        self.col_data[g*col_offset:(g+1)*col_offset])
                ).reshape(top_data[g*top_offset:(g+1)*top_offset].shape)

            if self.bias_term:
                for output_data, bias in zip(top_data, self.bias):
                    output_data += bias
        # out_groups = top.shape[1] // self.group
        # in_groups = bottom.shape[1] // self.group
        # for i in range(len(top)):
        #     for group in range(self.group):
        #         for out_group in range(out_groups):
        #             for in_group in range(in_groups):
        #                 convolve(
        #                     bottom[i, in_group + group * in_groups],
        #                     self.weights[out_group + group * out_groups,
        #                                  in_group],
        #                     top[i, out_group + group * out_groups],
        #                     self.padding, self.stride)
        #     if self.bias_term:
        #         for j in range(len(self.bias)):
        #             # TODO: Add support for sugar
        #             # top[i, j] += self.bias[j]
        #             top[i, j] += self.bias[j]

    def backward(self, top_data, top_diff, bottom_data, bottom_diff):
        if self.propagate_down:
            self.weight_diff.fill(0)
            if self.bias is not None:
                self.bias_diff.fill(0)

        for i in range(top_data.shape[0]):
            if self.bias is not None and self.propagate_down:
                curr_bias_diff = self.bias_diff[i]
                for n in range(top_data.shape[1]):
                    for data, bias_diff in zip(top_data[i, n], curr_bias_diff):
                        top_data += bias_diff
            if self.propagate_down:
                for n in range(top_data.shape[1]):
                    self.col_data = self.im2col(bottom_data, self.kernel_size,
                                                self.padding, self.stride)
                    col_offset = self.col_data.shape[0] // self.group
                    weight_offset = top_data.shape[0] // self.group
                    top_offset = top_data.shape[0] // self.group
                    for g in range(self.group):
                        self.weight_diff[i, g*weight_offset:(g+1)*weight_offset] += \
                            top_diff[i, g*top_offset:(g+1)*top_offset].dot(
                                self.col_data[
                                    i, g*col_offset:(g+1)*col_offset])

                    for g in range(self.group):
                        self.col_data[g*col_offset:(g+1)*col_offset] = \
                            self.weights[
                                g*weight_offset:(g+1)*weight_offset].dot(
                                    self.top_diff[
                                        n, g*top_offset:(g+1)*top_offset])
                    bottom_diff[i] = self.col2im(self.col_data,
                                                 self.kernel_size,
                                                 self.padding, self.stride)
