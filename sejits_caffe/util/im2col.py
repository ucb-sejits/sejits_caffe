import ctree.c.nodes as C
# import ctree.c
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import ctypes as ct
from cstructures.array import Array
import numpy as np


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
