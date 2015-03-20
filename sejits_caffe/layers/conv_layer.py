from .base_layer import BaseLayer
import numpy as np
import logging
# from sejits_caffe.util.im2col import cpu_im2col, gpu_im2col
from sejits_caffe.operations import convolve, meta
from cstructures.array import Array, specialize
import ctree.c.nodes as C
import ctree.c
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import ctypes as ct

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


class ConcreteIm2Col(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, out_shape):
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.out_shape = out_shape

    def __call__(self, *args):
        output = Array.empty(self.out_shape, np.float32)
        self._c_function(args[0], output)
        return output


class Im2Col(LazySpecializedFunction):
    def __init__(self, kernel_size, stride, padding):
        super(Im2Col, self).__init__(C.Constant(0))
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding

    def args_to_subconfig(self, args):
        A = args[0]
        return (A.shape, np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape))

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        channels, height, width = arg_cfg[0]
        cfg = {
            'pad_h': C.Constant(self.pad_h),
            'pad_w': C.Constant(self.pad_w),
            'stride_h': C.Constant(self.stride_h),
            'stride_w': C.Constant(self.stride_w),
            'kernel_h': C.Constant(self.kernel_h),
            'kernel_w': C.Constant(self.kernel_w),
            'channels': C.Constant(channels),
            'height': C.Constant(height),
            'width': C.Constant(width),
        }
        im2col = C.FunctionDecl(
            None,
            C.SymbolRef("im2col"),
            [C.SymbolRef("data_im", arg_cfg[1]()),
             C.SymbolRef("data_col", arg_cfg[1]())],
            [StringTemplate("""
int stride_h = $stride_h;
int stride_w = $stride_w;
int pad_h = $pad_h;
int pad_w = $pad_w;
int kernel_h = $kernel_h;
int kernel_w = $kernel_w;
int channels = $channels;
int height = $height;
int width = $width;
int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
int channels_col = channels * kernel_h * kernel_w;
for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
            int h_pad = h * stride_h - pad_h + h_offset;
            int w_pad = w * stride_w - pad_w + w_offset;
            if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            data_col[(c * height_col + h) * width_col + w] =
                data_im[(c_im * height + h_pad) * width + w_pad];
            else
                data_col[(c * height_col + h) * width_col + w] = 0;
        }
    }
} """, cfg)])
        return [C.CFile('im2col', [im2col])]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        channels, height, width = arg_cfg[0]
        height_col = (height + 2 * self.pad_h - self.kernel_h) // \
            self.stride_h + 1
        width_col = (width + 2 * self.pad_w - self.kernel_w) // \
            self.stride_w + 1
        out_shape = (channels * self.kernel_h * self.kernel_w, height_col *
                     width_col)
        out_ptr = np.ctypeslib.ndpointer(arg_cfg[1]._dtype_, 2, out_shape)
        entry_type = ct.CFUNCTYPE(None, arg_cfg[1], out_ptr)
        return ConcreteIm2Col('im2col', proj, entry_type, out_shape)


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
            else:
                raise Exception("Filler not implemented for weight filler \
                    type {}".format(weight_filler.type))
            if self.bias_term:
                self.bias = Array((num_output, ), np.float32)
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
            col_data = self.im2col(
                bottom_data, self.kernel_size, self.padding, self.stride)
            col_offset = col_data.shape[0] // self.group
            weight_offset = top_data.shape[0] // self.group
            top_offset = top_data.shape[0] // self.group
            for g in range(self.group):
                top_data[g*top_offset:(g+1)*top_offset] = (
                    weights[g*weight_offset:(g+1)*weight_offset].dot(
                        col_data[g*col_offset:(g+1)*col_offset])
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
