from .base_layer import BaseLayer
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.ocl import get_context_and_queue_from_devices
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import FunctionDecl, SymbolRef, CFile, Constant
from ctree.ocl.nodes import OclFile
from ctree.nodes import Project
import numpy as np
import logging
import pycl as cl
import ctypes as ct
from hindemith.nodes import kernel_range
from hindemith.types.hmarray import hmarray
from scipy.linalg.blas import dgemm


im2col_kernel = """
    int w_out = loop_idx % $width_col;
    int h_index = loop_idx / $width_col;
    int h_out = h_index % $height_col;
    int channel_in = h_index / $height_col;
    int channel_out = channel_in * $kernel_h * $kernel_w;
    int h_in = h_out * $stride_h - $pad_h;
    int w_in = w_out * $stride_w - $pad_w;
    __global $Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * $height_col + h_out) * $width_col + w_out;
    __global $Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * $height + h_in) * $width + w_in;
    for (int i = 0; i < $kernel_h; ++i) {
      for (int j = 0; j < $kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < $height && w < $width) ?
            data_im_ptr[i * $width + j] : 0;
        data_col_ptr += $height_col * $width_col;
      }
    }
"""


class OclConcreteIm2Col(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, *args):
        output = None
        out_buf = None
        channels, height, width = args[1]
        kernel_height, kernel_width = args[2]
        padding_height, padding_width = args[3]
        stride_height, stride_width = args[4]
        height_out_ = (height + 2 * padding_height - kernel_height) / stride_height + 1
        width_out_ = (width + 2 * padding_width - kernel_width) / stride_width + 1
        output = hmarray(np.zeros((channels * kernel_height * kernel_width, height_out_, width_out_), np.float32))
        out_buf = cl.clCreateBuffer(self.context, output.nbytes)
        output._ocl_buf = out_buf
        output._ocl_dirty = False
        output._host_dirty = True
        self._c_function(*([self.queue, self.kernel, args[0].ocl_buf, out_buf]))
        return output


class Im2Col(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        channels, height, width = args[1]
        kernel_height, kernel_width = args[2]
        padding_height, padding_width = args[3]
        stride_height, stride_width = args[4]
        return {
            'ptr': np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape),
            'channels': channels,
            'height': height,
            'width': width,
            'kernel_height': kernel_height,
            'kernel_width': kernel_width,
            'padding_height': padding_height,
            'padding_width': padding_width,
            'stride_height': stride_height,
            'stride_width': stride_width
        }

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        height_col = (arg_cfg['height'] + 2 * arg_cfg['padding_height'] -
                      arg_cfg['kernel_height']) / arg_cfg['stride_height'] + 1
        width_col = (arg_cfg['width'] + 2 * arg_cfg['padding_width'] -
                     arg_cfg['kernel_width']) / arg_cfg['stride_width'] + 1
        num_kernels = arg_cfg['channels'] * height_col * width_col
        loop_body = [StringTemplate(im2col_kernel, {
            'Dtype': SymbolRef('float'),
            'num_kernels': Constant(num_kernels),
            'height': Constant(arg_cfg['height']),
            'width': Constant(arg_cfg['width']),
            'kernel_h': Constant(arg_cfg['kernel_height']),
            'kernel_w': Constant(arg_cfg['kernel_width']),
            'pad_h': Constant(arg_cfg['padding_height']),
            'pad_w': Constant(arg_cfg['padding_width']),
            'stride_h': Constant(arg_cfg['stride_height']),
            'stride_w': Constant(arg_cfg['stride_width']),
            'height_col': Constant(height_col),
            'width_col': Constant(width_col),
            })]

        kernel_params = [SymbolRef("data_im", arg_cfg['ptr']()),
                         SymbolRef("data_col", arg_cfg['ptr']())]
        func = FunctionDecl(
            None,
            SymbolRef('im2col'),
            [SymbolRef("data_im", cl.cl_mem()),
             SymbolRef("data_col", cl.cl_mem())],
            []
        )
        proj = Project([CFile('im2col', [func])])
        proj.files[0].body.insert(0, StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """))
        arg_types = (cl.cl_command_queue, cl.cl_kernel, cl.cl_mem, cl.cl_mem)
        shape = [arg_cfg['channels'], height_col, width_col]
        control, kernel = kernel_range(shape, shape,
                                       kernel_params, loop_body)
        func.defn = control
        func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
        func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                        cl.cl_kernel()))
        proj.files.append(kernel)
        entry_type = (None,) + arg_types
        return 'im2col', proj, entry_type

    def finalize(self, entry_name, proj, entry_type):
        entry_type = ct.CFUNCTYPE(*entry_type)
        fn = OclConcreteIm2Col(entry_name, proj, entry_type)
        kernel = proj.find(OclFile)
        program = cl.clCreateProgramWithSource(
            fn.context, kernel.codegen()).build()
        return fn.finalize(program[kernel.body[0].name.name])


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
        self.group = group
        assert channels % group == 0, \
            "Number of channels should be a multiple of group."
        assert num_output % group == 0, \
            "Number of outputs should be a multiple of group."

        self.M = num_output / group
        self.K = channels * np.prod(kernel_size) / group
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
        weights = self.blobs[0]

        for bottom_data, top_data in zip(bottom, top):
            weight_offset = self.M * self.K
            col_offset = self.K * self.N
            top_offset = self.M * self.N
            for n in range(len(bottom_data)):
                col_data = im2col(bottom_data, bottom_data.shape,
                                  (self.kernel_size, self.kernel_size),
                                  (self.padding, self.padding),
                                  (self.stride, self.stride))
                for g in range(self.group):
                    print(self.M)
                    print(self.N)
                    print(self.K)
                    print(col_data.shape)
                    print(weights.shape)
                    np.dot(col_data[g], weights, top_data[g])

                    if self.bias_term:
                        np.dot(self.blobs[1], self.bias_multiplier, top_data)
