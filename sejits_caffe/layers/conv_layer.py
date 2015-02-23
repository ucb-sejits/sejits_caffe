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

        self.height_out = (height + 2 * self.pad_h - self.kernel_h) // \
            self.stride_h + 1
        self.width_out = (width + 2 * self.pad_w - self.kernel_w) // \
            self.stride_w + 1

        self.conv_out_channels = num_output
        self.conv_in_channels = channels
        self.conv_in_height = height
        self.conv_in_width = width
        self.conv_out_spatial_dim = self.height_out * self.width_out
        self.kernel_dim = self.conv_in_channels * self.kernel_h * self.kernel_w
        self.weight_offset = self.conv_out_channels * self.kernel_dim // self.group \
            // self.group
        self.col_offset = self.kernel_dim * self.conv_out_spatial_dim // \
            self.group
        self.output_offset = self.conv_out_channels * self.conv_out_spatial_dim \
            // self.group

        self.bias_term = conv_param.bias_term
        if hasattr(self, 'weights'):
            logging.debug("Skipping parameter initialization")
        else:
            self.weights = hmarray((num_output, channels // self.group,
                                    self.kernel_h, self.kernel_w), np.float32)
            self.weights.fill(.1)
            self.weights._ocl_dirty = True
            if self.bias_term:
                self.bias = np.ndarray((num_output, ), np.float32)
                filler = conv_param.bias_filler
                if filler.type == 'constant':
                    self.bias.fill(filler.value)
                else:
                    raise Exception("Filler not implemented for bias filler \
                        type {}".format(filler.type))

    def cpu_forward(self, bottom, top):
        for bottom_data, top_data in zip(bottom, top):
            col_data = cpu_im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_h, self.kernel_w),
                                  (self.pad_h, self.pad_w),
                                  (self.stride_h, self.stride_w))
            # TODO: Add support for group > 1
            for g in range(self.group):
                cpu_gemm(self.weights, g * self.weight_offset,
                         col_data, g * self.col_offset,
                         top_data, g * self.output_offset,
                         self.conv_out_channels // self.group,
                         self.conv_out_spatial_dim,
                         self.kernel_dim // self.group)

            if self.bias_term:
                top_data += self.bias[:, np.newaxis]

    def gpu_forward(self, bottom, top):
        for bottom_data, top_data in zip(bottom, top):
            col_data = gpu_im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_h, self.kernel_w),
                                  (self.pad_h, self.pad_w),
                                  (self.stride_h, self.stride_w))

            # TODO: Add support for group > 1
            for g in range(self.group):
                gpu_gemm(self.weights, g * self.weight_offset,
                         col_data, g * self.col_offset,
                         top_data, g * self.output_offset,
                         self.conv_out_channels // self.group,
                         self.conv_out_spatial_dim,
                         self.kernel_dim // self.group)
            top_data.copy_to_host_if_dirty()

            if self.bias_term:
                top_data += self.bias[:, np.newaxis]

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
