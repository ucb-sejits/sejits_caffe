from .base_layer import BaseLayer
import numpy as np
import logging
from sejits_caffe.util.im2col import cpu_im2col, gpu_im2col
from hindemith import hmarray
# from hindemith.operations.gemm import gemm
import ctypes
from ctypes import c_int, c_float, c_size_t
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


def cpu_gemm(A, B, C, m, n, k):

    cblas_row_major = c_int(101)
    no_trans = c_int(111)
    m = c_int(int(m))
    n = c_int(int(n))
    k = c_int(int(k))
    one = c_float(1.0)
    zero = c_float(0.0)

    _blaslib.cblas_sgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                         one, A.ctypes.data_as(ctypes.c_void_p), k,
                         B.ctypes.data_as(ctypes.c_void_p), n, zero,
                         C.ctypes.data_as(ctypes.c_void_p), n)


from ctree.ocl import get_context_and_queue_from_devices
devices = cl.clGetDeviceIDs()
context, queue = get_context_and_queue_from_devices([devices[-1]])
err = _clblaslib.clblasSetup()


def gpu_gemm(A, B, C, m, n, k):
    cblas_row_major = c_int(0)
    no_trans = c_int(0)
    m = c_size_t(int(m))
    n = c_size_t(int(n))
    k = c_size_t(int(k))
    one = c_float(1.0)
    zero = c_float(0.0)

    cl.clFinish(queue)
    A.copy_to_host_if_dirty()
    B.copy_to_host_if_dirty()
    C.copy_to_host_if_dirty()
    a_buf, evt = cl.buffer_from_ndarray(queue, A)
    evt.wait()
    b_buf, evt = cl.buffer_from_ndarray(queue, B)
    evt.wait()
    c_buf, evt = cl.buffer_from_ndarray(queue, C)
    evt.wait()
    cl.clFinish(queue)
    err = _clblaslib.clblasSgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                                 one, a_buf, c_size_t(0), k, b_buf,
                                 c_size_t(0), n, zero, c_buf, c_size_t(0), n,
                                 c_size_t(1), ctypes.byref(queue), c_size_t(0),
                                 None, None)
    print(err)
    C._ocl_buf = c_buf
    C._host_dirty = True
    cl.clFinish(queue)


class ConvLayer(BaseLayer):
    backend = 'gpu'

    def set_up(self, bottom, top):
        conv_param = self.layer_param.convolution_param

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

        channels, height, width = bottom[0].shape
        num_output = conv_param.num_output
        self.group = conv_param.group
        assert channels % self.group == 0, \
            "Number of channels should be a multiple of group."
        assert num_output % self.group == 0, \
            "Number of outputs should be a multiple of group."

        self.M = num_output // self.group
        self.K = channels * self.kernel_h * self.kernel_w // self.group
        self.height_out = (height + 2 * self.pad_h - self.kernel_h) // \
            self.stride_h + 1
        self.width_out = (width + 2 * self.pad_w - self.kernel_w) // \
            self.stride_w + 1
        self.N = self.height_out * self.width_out

        self.bias_term = conv_param.bias_term
        if hasattr(self, 'weights'):
            logging.debug("Skipping parameter initialization")
        else:
            self.weights = hmarray((self.M, self.K), np.float32)
            self.weights.fill(.1)
            self.weights._ocl_dirty = True
            if self.bias_term:
                self.bias = np.ndarray((num_output, ))
                self.bias.fill(0)

    def cpu_forward(self, bottom, top):
        for bottom_data, top_data in zip(bottom, top):
            col_data = cpu_im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_h, self.kernel_w),
                                  (self.pad_h, self.pad_w),
                                  (self.stride_h, self.stride_w))
            # TODO: Add support for group > 1
            # for g in range(self.group):

            cpu_gemm(self.weights, col_data, top_data, self.M, self.N, self.K)

            if self.bias_term:
                top_data += self.bias[:, np.newaxis]

    def gpu_forward(self, bottom, top):
        for bottom_data, top_data in zip(bottom, top):
            col_data = gpu_im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_h, self.kernel_w),
                                  (self.pad_h, self.pad_w),
                                  (self.stride_h, self.stride_w))

            # TODO: Add support for group > 1
            # for g in range(self.group):

            gpu_gemm(self.weights, col_data, top_data, self.M, self.N, self.K)
            top_data.copy_to_host_if_dirty()

            # if self.bias_term:
            #     top_data += self.bias[:, np.newaxis]

    def forward(self, bottom, top):
        if self.backend == 'gpu':
            self.gpu_forward(bottom, top)
        else:
            self.cpu_forward(bottom, top)

    # def backward(self, top, propagate_down, bottom):
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
