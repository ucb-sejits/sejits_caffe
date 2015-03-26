import ctree.c.nodes as C
# import ctree.c
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import ctypes as ct
from cstructures.array import Array
import numpy as np


class ConcreteCol2Im(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, out_shape):
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.out_shape = out_shape

    def __call__(self, *args):
        output = Array.zeros(self.out_shape, args[0].dtype)
        self._c_function(args[0], output)
        return output


class Col2Im(LazySpecializedFunction):
    def __init__(self, kernel_size, stride, padding, shape):
        super(Col2Im, self).__init__(C.Constant(0))
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding
        self.shape = shape

    def args_to_subconfig(self, args):
        A = args[0]
        return (A.shape, np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape))

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        channels, height, width = self.shape
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
        col2im = C.FunctionDecl(
            None,
            C.SymbolRef("col2im"),
            [C.SymbolRef("data_col", arg_cfg[1]()),
             C.SymbolRef("data_im", arg_cfg[1]())],
            [StringTemplate("""
int stride_h = $stride_h;
int stride_w = $stride_w;
int pad_h = $pad_h;
int pad_w = $pad_w;
int patch_h = $kernel_h;
int patch_w = $kernel_w;
int channels = $channels;
int height = $height;
int width = $width;
int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
int channels_col = channels * patch_h * patch_w;
for (int c = 0; c < channels_col; ++c) {
  int w_offset = c % patch_w;
  int h_offset = (c / patch_w) % patch_h;
  int c_im = c / patch_h / patch_w;
  for (int h = 0; h < height_col; ++h) {
    for (int w = 0; w < width_col; ++w) {
      int h_pad = h * stride_h - pad_h + h_offset;
      int w_pad = w * stride_w - pad_w + w_offset;
      if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
        data_im[(c_im * height + h_pad) * width + w_pad] +=
            data_col[(c * height_col + h) * width_col + w];
    }
  }
} """, cfg)])
        return [C.CFile('col2im', [col2im])]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        out_ptr = np.ctypeslib.ndpointer(arg_cfg[1]._dtype_, 3, self.shape)
        entry_type = ct.CFUNCTYPE(None, arg_cfg[1], out_ptr)
        return ConcreteCol2Im('col2im', proj, entry_type, self.shape)
