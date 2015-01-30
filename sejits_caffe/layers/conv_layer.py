from base_layer import BaseLayer
import numpy as np
import logging
# import pycl as cl
from sejits_caffe.blob import Blob
from sejits_caffe.util.im2col import im2col


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
