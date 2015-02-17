# from ctree.ocl.nodes import OclFile
# from ctree.ocl import get_context_and_queue_from_devices
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import FunctionDecl, SymbolRef, CFile, Constant
from ctree.ocl.nodes import OclFile
from ctree.ocl import get_context_and_queue_from_devices
from ctree.nodes import Project
import ctypes as ct
from hindemith.types.hmarray import empty
from hindemith.nodes import kernel_range
import numpy as np
import pycl as cl


im2col_c = """
  for (int c = 0; c < $channels_col; ++c) {
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

im2col_ocl = """
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
        out_shape = (channels * kernel_h * kernel_w, h_out * w_out)
        output = empty(out_shape, np.float32)
        self._c_function(args[0], output)
        return output


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
        channels, h, w = args[1]
        kernel_h, kernel_w = args[2]
        padding_h, padding_w = args[3]
        stride_h, stride_w = args[4]
        h_out = (h + 2 * padding_h - kernel_h) / stride_h + 1
        w_out = (w + 2 * padding_w - kernel_w) / stride_w + 1
        out_shape = (channels * kernel_h * kernel_w, h_out * w_out)
        output = empty(out_shape, np.float32)
        output._host_dirty = True
        self._c_function(self.queue, self.kernel, args[0].ocl_buf,
                         output.ocl_buf)
        return output


import ast


class Im2Col(LazySpecializedFunction):
    def __init__(self, backend='c'):
        super(Im2Col, self).__init__(ast.Module())
        self.backend = backend

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
            'stride_w': stride_w,
            'backend': self.backend
        }

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        channels_col = arg_cfg['channels'] * arg_cfg['kernel_h'] * \
            arg_cfg['kernel_w']
        height_col = (arg_cfg['height'] + 2 * arg_cfg['padding_h'] -
                      arg_cfg['kernel_h']) // arg_cfg['stride_h'] + 1
        width_col = (arg_cfg['width'] + 2 * arg_cfg['padding_w'] -
                     arg_cfg['kernel_w']) // arg_cfg['stride_w'] + 1
        out_shape = (arg_cfg['channels'] * arg_cfg['kernel_h'] *
                     arg_cfg['kernel_w'], height_col * width_col)
        out_ptr = np.ctypeslib.ndpointer(arg_cfg['ptr']._dtype_, 2, out_shape)
        if self.backend == 'c':
            loop_body = [StringTemplate(im2col_c, {
                'Dtype': SymbolRef('float'),
                'channels': Constant(arg_cfg['channels']),
                'height': Constant(arg_cfg['height']),
                'width': Constant(arg_cfg['width']),
                'kernel_h': Constant(arg_cfg['kernel_h']),
                'kernel_w': Constant(arg_cfg['kernel_w']),
                'pad_h': Constant(arg_cfg['padding_h']),
                'pad_w': Constant(arg_cfg['padding_w']),
                'stride_h': Constant(arg_cfg['stride_h']),
                'stride_w': Constant(arg_cfg['stride_w']),
                'channels_col': Constant(channels_col),
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
            func.defn = loop_body
            entry_type = (None, arg_cfg['ptr'], out_ptr)
        elif self.backend == 'ocl':
            loop_body = [StringTemplate(im2col_ocl, {
                'Dtype': SymbolRef('float'),
                'channels': Constant(arg_cfg['channels']),
                'height': Constant(arg_cfg['height']),
                'width': Constant(arg_cfg['width']),
                'kernel_h': Constant(arg_cfg['kernel_h']),
                'kernel_w': Constant(arg_cfg['kernel_w']),
                'pad_h': Constant(arg_cfg['padding_h']),
                'pad_w': Constant(arg_cfg['padding_w']),
                'stride_h': Constant(arg_cfg['stride_h']),
                'stride_w': Constant(arg_cfg['stride_w']),
                'channels_col': Constant(channels_col),
                'height_col': Constant(height_col),
                'width_col': Constant(width_col),
                })]
            shape = (arg_cfg['channels'] * height_col * width_col,)
            params = [SymbolRef("data_im", arg_cfg['ptr']()),
                      SymbolRef("data_col", out_ptr())]
            control, kernel = kernel_range(shape, shape, params, loop_body)
            func = FunctionDecl(
                None,
                SymbolRef('im2col'),
                [SymbolRef("data_im", cl.cl_mem()),
                 SymbolRef("data_col", cl.cl_mem())],
                control
            )
            func.params.insert(0, SymbolRef('queue', cl.cl_command_queue()))
            func.params.insert(1, SymbolRef(kernel.body[0].name.name,
                                            cl.cl_kernel()))
            proj = Project([CFile('im2col', [func]), kernel])
            proj.files[0].config_target = 'opencl'
            proj.files[0].body.insert(0, StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
                """))
            shape = [arg_cfg['channels'], height_col, width_col]
            entry_type = (None, cl.cl_command_queue, cl.cl_kernel,
                          cl.cl_mem, cl.cl_mem)
        else:
            raise Exception(
                "Unsupport backend for im2col {}".format(self.backend))
        return proj.files

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg

        height_col = (arg_cfg['height'] + 2 * arg_cfg['padding_h'] -
                      arg_cfg['kernel_h']) // arg_cfg['stride_h'] + 1
        width_col = (arg_cfg['width'] + 2 * arg_cfg['padding_w'] -
                     arg_cfg['kernel_w']) // arg_cfg['stride_w'] + 1
        out_shape = (arg_cfg['channels'] * arg_cfg['kernel_h'] *
                     arg_cfg['kernel_w'], height_col * width_col)
        out_ptr = np.ctypeslib.ndpointer(arg_cfg['ptr']._dtype_, 2, out_shape)
        proj = Project(files)
        if self.backend == 'c':
            entry_type = (None, arg_cfg['ptr'], out_ptr)
            entry_type = ct.CFUNCTYPE(*entry_type)
            fn = CConcreteIm2Col('im2col', proj, entry_type)
            return fn
        elif self.backend == 'ocl':
            entry_type = (None, cl.cl_command_queue, cl.cl_kernel,
                          cl.cl_mem, cl.cl_mem)
            entry_type = ct.CFUNCTYPE(*entry_type)
            fn = OclConcreteIm2Col('im2col', proj, entry_type)
            kernel = proj.find(OclFile)
            program = cl.clCreateProgramWithSource(
                fn.context, kernel.codegen()).build()
            return fn.finalize(program[kernel.name])

cpu_im2col = Im2Col()
gpu_im2col = Im2Col(backend='ocl')
