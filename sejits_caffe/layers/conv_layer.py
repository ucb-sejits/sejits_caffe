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
    def __init__(self, blobs, bottom, top, num_output, kernel_size, padding=0,
                 stride=1, group=1, bias_term=True):
        """
        :param int kernel_size:
        Square filter size
        TODO: Support non square filters

        :param int stride:
        The filter stride, default a dense convolution of stride 1.
        TODO: Suppport different strides in height and width direction

        :param int pad:
        The zero-padding for convolution
        TODO: Support non symmetric padding

        :param int group: optional, default 1.
        The number of filter groups. Group convolution is a method
        for reducing paramaterization by selectively connecting input
        and output channels.  The input and outpuyt channel dimension
        must be divisible by the number of groups. For group f >= 1,
        the convolutional filters' input and output channels are
        separated s.t. each group takes 1 / group of the input
        channels 1-2 and output channels 1-4 into the first group and
        input channels 3-4 and output channels 5-8 into the second
        group.

        :param bool bias_term: optional, default True.
        Whether to have a bias.
        """
        self.blobs = blobs
        assert kernel_size > 0, "Filter dimensions cannot be zero."
        self.kernel_size = kernel_size

        self.padding = padding
        self.stride = stride

        assert num_output > 0, "Layer must have at least one output"

        channels, height, width = bottom[0].shape
        self.channels = channels
        self.group = group
        assert channels % group == 0, \
            "Number of channels should be a multiple of group."
        assert num_output % group == 0, \
            "Number of outputs should be a multiple of group."

        self.M = num_output / group
        self.K = channels * kernel_size * kernel_size / group
        self.height_out = (height + 2 * padding - kernel_size) / stride + 1
        self.width_out = (width + 2 * padding - kernel_size) / stride + 1
        self.N = self.height_out * self.width_out

        self.bias_term = bias_term
        if len(self.blobs) > 0:
            logging.debug("Skipping parameter initialization")
        else:
            if bias_term:
                self.blobs.resize(2)
            else:
                self.blobs.resize(1)

    def forward(self, bottom, top):
        blob = self.blobs[0]
        blob_length = np.prod(blob.shape)
        weight_shape = (self.M, self.channels * blob_length)
        weights = np.ndarray(weight_shape, np.float32)
        blob = blob.reshape(blob_length)
        for i in range(self.channels):
            weights[..., i*blob_length:(i+1)*blob_length] = blob

        for bottom_data, top_data in zip(bottom, top):
            for n in range(len(bottom_data)):
                col_data = im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_size, self.kernel_size),
                                  (self.padding, self.padding),
                                  (self.stride, self.stride))
                shape = col_data.shape
                data = col_data.reshape((shape[0], np.prod(shape[1:])))
                out = np.dot(weights, data)
                top_data[:] = out[:]

                # for g in range(self.group):
                #     if self.bias_term:
                #         np.dot(self.blobs[1], self.bias_multiplier, top_data)
