from .base_layer import BaseLayer
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import FunctionDecl, SymbolRef, CFile, Constant
# from ctree.ocl.nodes import OclFile
# from ctree.ocl import get_context_and_queue_from_devices
from ctree.nodes import Project
import numpy as np
import logging
# import pycl as cl
import ctypes as ct
from ..blob import Blob
from hindemith.types.hmarray import empty


im2col_kernel = """
  int channels_col = $channels * $kernel_h * $kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % $kernel_w;
    int h_offset = (c / $kernel_w) % $kernel_h;
    int c_im = c / $kernel_h / $kernel_w;
    for (int h = 0; h < $height_col; ++h) {
      for (int w = 0; w < $width_col; ++w) {
        int h_pad = h * $stride_h - $pad_h + h_offset;
        int w_pad = w * $stride_w - $pad_w + w_offset;
        if (h_pad >= 0 && h_pad < $height && w_pad >= 0 && w_pad < $width)
          data_col[(c * $height_col + h) * $width_col + w] =
            data_im[(c_im * $height + h_pad) * $width + w_pad];
        else
          data_col[(c * $height_col + h) * $width_col + w] = 0;
      }
    }
  }
"""


class CConcreteIm2Col(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args):
        output = None
        channels, h, w = args[1]
        kernel_h, kernel_w = args[2]
        padding_h, padding_w = args[3]
        stride_h, stride_w = args[4]
        h_out = (h + 2 * padding_h - kernel_h) / stride_h + 1
        w_out = (w + 2 * padding_w - kernel_w) / stride_w + 1
        out_shape = (channels * kernel_h * kernel_w, h_out, w_out)
        output = empty(out_shape, np.float32)
        self._c_function(args[0], output)
        return output


class Im2Col(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        channels, height, width = args[1]
        kernel_h, kernel_w = args[2]
        padding_h, padding_w = args[3]
        stride_h, stride_w = args[4]
        return {
            'ptr': np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape),
            'channels': channels,
            'height': height,
            'width': width,
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'padding_h': padding_h,
            'padding_w': padding_w,
            'stride_h': stride_h,
            'stride_w': stride_w
        }

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        height_col = (arg_cfg['height'] + 2 * arg_cfg['padding_h'] -
                      arg_cfg['kernel_h']) / arg_cfg['stride_h'] + 1
        width_col = (arg_cfg['width'] + 2 * arg_cfg['padding_w'] -
                     arg_cfg['kernel_w']) / arg_cfg['stride_w'] + 1
        out_shape = (arg_cfg['channels'] * arg_cfg['kernel_h'] *
                     arg_cfg['kernel_w'], height_col, width_col)
        out_ptr = np.ctypeslib.ndpointer(arg_cfg['ptr']._dtype_, 3, out_shape)
        num_kernels = arg_cfg['channels'] * height_col * width_col
        loop_body = [StringTemplate(im2col_kernel, {
            'Dtype': SymbolRef('float'),
            'num_kernels': Constant(num_kernels),
            'channels': Constant(arg_cfg['channels']),
            'height': Constant(arg_cfg['height']),
            'width': Constant(arg_cfg['width']),
            'kernel_h': Constant(arg_cfg['kernel_h']),
            'kernel_w': Constant(arg_cfg['kernel_w']),
            'pad_h': Constant(arg_cfg['padding_h']),
            'pad_w': Constant(arg_cfg['padding_w']),
            'stride_h': Constant(arg_cfg['stride_h']),
            'stride_w': Constant(arg_cfg['stride_w']),
            'height_col': Constant(height_col),
            'width_col': Constant(width_col),
            })]

        func = FunctionDecl(
            None,
            SymbolRef('im2col'),
            [SymbolRef("data_im", arg_cfg['ptr']()),
             SymbolRef("data_col", out_ptr())],
            []
        )
        proj = Project([CFile('im2col', [func])])
        # proj.files[0].body.insert(0, StringTemplate("""
        #     #ifdef __APPLE__
        #     #include <OpenCL/opencl.h>
        #     #else
        #     #include <CL/cl.h>
        #     #endif
        #     """) )
        # arg_types = (cl.cl_command_queue, cl.cl_kernel, cl.cl_mem, cl.cl_mem)
        # shape = [arg_cfg['channels'], height_col, width_col]
        func.defn = [loop_body]
        entry_type = (None, arg_cfg['ptr'], out_ptr)
        return 'im2col', proj, entry_type

    def finalize(self, entry_name, proj, entry_type):
        entry_type = ct.CFUNCTYPE(*entry_type)
        fn = CConcreteIm2Col(entry_name, proj, entry_type)
        return fn


im2col = Im2Col(None)


class ConvLayer(BaseLayer):
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

        self.M = num_output / self.group
        self.K = channels * self.kernel_h * self.kernel_w / self.group
        self.height_out = (height + 2 * self.pad_h - self.kernel_h) / \
            self.stride_h + 1
        self.width_out = (width + 2 * self.pad_w - self.kernel_w) / \
            self.stride_w + 1
        self.N = self.height_out * self.width_out

        self.bias_term = conv_param.bias_term
        if len(self.blobs) > 0:
            logging.debug("Skipping parameter initialization")
        else:
            self.blobs.append(Blob(num_output, channels / self.group,
                                   self.kernel_h, self.kernel_w))
            self.blobs[0].fill(.1)
            if self.bias_term:
                self.blobs.append(Blob(1, 1, 1, num_output))
                self.blobs[1].fill(0)

    def forward(self, bottom, top):
        weights = self.blobs[0].data.reshape((self.M, self.K))
        for bottom_data, top_data in zip(bottom, top):
            for n in range(len(bottom_data)):
                col_data = im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_h, self.kernel_w),
                                  (self.pad_h, self.pad_w),
                                  (self.stride_h, self.stride_w))
                data = col_data.reshape((self.K, self.N))

                # TODO: Add support for group > 1
                # for g in range(self.group):

                # TODO: Weirdness in reshape method prevents us from doing dot
                # directly into the output.  Should initialize the arrays with
                # the right shape so we don't have to call reshape
                top_data[:] = np.dot(weights, data)

                if self.bias_term:
                    top_data += self.blobs[1].data[:, np.newaxis]

    def backward(self, top, propagate_down, bottom):
        weight = None
        weight_diff = None
        if self.param_propagate_down[0]:
            weight = self.blobs[0].data
            weight_diff = self.blobs[0].diff
            weight_diff[...] = 0

        bias_diff = None
        if self.bias_term and self.param_propagate_down[1]:
            bias_diff = self.blobs[1].diff
            bias_diff[...] = 0

        for top_data, bottom_data, prop in zip(top, bottom, propagate_down):
            top_diff = None
            if self.bias_term and self.param_propagate_down[1]:
                top_diff = top_data.diff
                for n in range(self.num):
                    bias_diff += top_diff

            if self.param_propagate_down[0] or prop:
                if not top_diff:
                    top_diff = top_data.diff
                for n in range(self.num):
                    col_data = im2col(bottom_data, bottom_data.shape,
                                      (self.kernel_size, self.kernel_size),
                                      (self.padding, self.padding),
                                      (self.stride, self.stride))
                    top_data = top_data.reshape((self.M, self.K))
                    data = col_data.reshape((self.K, self.N))
                    if self.param_propagate_down[0]:
                        weight += np.dot(top_data, data)

                    if prop:
                        if weight is None:
                            weight = self.blobs[0].data
                        weight = weight.reshape((self.M, self.K))
                        top_data = top_data.reshape((self.K, self.N))
                        col_data[:] = np.dot(weight, top_data)

