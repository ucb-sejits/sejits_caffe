from .base_layer import BaseLayer
from ctree.jit import LazySpecializedFunction
from ctree.templates.nodes import StringTemplate
import numpy as np
import logging


im2col_kernel = StringTemplate(
    """
    int w_out = loop_idx % width_col;
    int h_index = loop_idx / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
    """
)


class Im2Col(LazySpecializedFunction):
    def __init__(self, channels, size, kernel_size, padding, stride):
        self.channels = channels
        self.size = size
        self.padding = padding
        self.stride = stride

    def args_to_subconfig(self, args):
        A = args[0]
        return (np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape), )

    def transform(self, tree, program_cfg):
        pass


class ConvLayer(BaseLayer):
    def __init__(self, bottom, top, num_output, kernel_size, padding=0,
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
        assert kernel_size > 0, "Filter dimensions cannot be zero."
        self.kernel_size = kernel_size

        self.padding = padding
        self.stride = stride

        assert num_output > 0, "Layer must have at least one output"

        channels = bottom[0].channels
        self.group = group
        assert channels % group == 0, \
            "Number of channels should be a multiple of group."
        assert num_output % group == 0, \
            "Number of outputs should be a multiple of group."
        self.bias_term = bias_term
        if len(self.blobs) > 0:
            logging.debug("Skipping parameter initialization")
        else:
            if bias_term:
                self.blobs.resize(2)
            else:
                self.blobs.resize(1)

        self.im2col = Im2Col(channels, bottom[0].size,
                             self.kernel_size, self.padding, self.stride)

    def forward(self, bottom, top):
        weights = self.blobs[0]

        for bottom_data, top_data in zip(bottom, top):
            for n in self.num:
                col_data = self.im2col(bottom_data)
                for g in self.group:
                    np.dot(col_data[g], weights[g], top_data[g])

                    if self.bias_term:
                        np.dot(self.blobs[1], self.bias_multiplier, top_data)
