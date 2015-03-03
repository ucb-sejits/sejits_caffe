from .base_layer import BaseLayer
import numpy as np
import logging
# from sejits_caffe.util.im2col import cpu_im2col, gpu_im2col
from sejits_caffe.operations import convolve, meta
from sejits_caffe.types import Array

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

        self.pad_h = conv_param.pad
        self.pad_w = conv_param.pad

        self.stride_h = conv_param.stride
        self.stride_w = conv_param.stride

        assert conv_param.num_output > 0, "Layer must have at least one output"

        self.group = conv_param.group

        self.conv_out_channels = None
        self.conv_in_channels = None
        self.conv_in_height = None
        self.conv_in_width = None
        self.conv_out_spatial_dim = None
        self.kernel_dim = None
        self.weight_offset = None
        self.col_offset = None
        self.output_offset = None
        self.weights = None
        self.bias_term = None
        self.bias = None

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
            else:
                raise Exception("Filler not implemented for weight filler \
                    type {}".format(weight_filler.type))
            if self.bias_term:
                # FIXME: This should be a 1d array
                self.bias = Array((num_output, top.shape[-2], top.shape[-1]),
                                  np.float32)
                filler = conv_param.bias_filler
                if filler.type == 'constant':
                    self.bias.fill(filler.value)
                else:
                    raise Exception("Filler not implemented for bias filler \
                        type {}".format(filler.type))

    @meta
    def forward(self, bottom, top):
        out_groups = top.shape[1] // self.group
        in_groups = bottom.shape[1] // self.group
        for i in range(len(top)):
            for group in range(self.group):
                for out_group in range(out_groups):
                    for in_group in range(in_groups):
                        convolve(
                            bottom[i, in_group + group * in_groups],
                            self.weights[out_group + group * out_groups,
                                         in_group],
                            top[i, out_group + group * out_groups],
                            (self.pad_h, self.pad_w),
                            (self.stride_h, self.stride_w))

            if self.bias_term:
                for j in range(len(self.bias)):
                    # TODO: Add support for sugar
                    # top[i, j] += self.bias[j]
                    Array.add(top[i, j], self.bias[j], top[i, j])

    def backward(self, top, propagate_down, bottom):
        pass
    #     weight = None
    #     weight_diff = None
    #     if self.param_propagate_down[0]:
    #         weight = self.blobs[0].data
    #         weight_diff = self.blobs[0].diff
    #         weight_diff[...] = 0

    #     bias_diff = None
    #     if self.bias_term and self.param_propagate_down[1]:
    #         bias_diff = self.blobs[1].diff
    #         bias_diff[...] = 0

    #     for top_data, bottom_data, prop in zip(top, bottom, propagate_down):
    #         top_diff = None
    #         if self.bias_term and self.param_propagate_down[1]:
    #             top_diff = top_data.diff
    #             for n in range(self.num):
    #                 bias_diff += top_diff

    #         if self.param_propagate_down[0] or prop:
    #             if not top_diff:
    #                 top_diff = top_data.diff
    #             for n in range(self.num):
    #                 col_data = im2col(bottom_data, bottom_data.shape,
    #                                   (self.kernel_size, self.kernel_size),
    #                                   (self.padding, self.padding),
    #                                   (self.stride, self.stride))
    #                 top_data = top_data.reshape((self.M, self.K))
    #                 data = col_data.reshape((self.K, self.N))
    #                 if self.param_propagate_down[0]:
    #                     weight += np.dot(top_data, data)

    #                 if prop:
    #                     if weight is None:
    #                         weight = self.blobs[0].data
    #                     weight = weight.reshape((self.M, self.K))
    #                     top_data = top_data.reshape((self.K, self.N))
    #                     col_data[:] = np.dot(weight, top_data)
