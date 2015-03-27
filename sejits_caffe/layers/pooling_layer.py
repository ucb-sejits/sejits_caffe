from cstructures import Array
from sejits_caffe.operations import max_pool  # , meta
from sejits_caffe.layers.base_layer import BaseLayer
from cstructures.array import specialize


@specialize
def max_pool_backward(top_diff, bottom_diff, mask):  # pragma: no cover
    for y, x in top_diff.indices():
        index = mask[y, x]
        bottom_diff[index] += top_diff[y, x]


class PoolingLayer(BaseLayer):
    """docstring for PoolingLayer"""
    def __init__(self, param):
        super(PoolingLayer, self).__init__(param)
        pool_param = param.pooling_param
        if pool_param.kernel_size is not None:
            self.kernel_h = self.kernel_w = pool_param.kernel_size
        elif pool_param.kernel_h is not None and \
                pool_param.kernel_w is not None:
            self.kernel_h = pool_param.kernel_h
            self.kernel_w = pool_param.kernel_w
        else:
            raise Exception("Pooling Layer must specify kernel size or"
                            "kernel_h and kernel_w")

        assert self.kernel_h > 0 and self.kernel_w > 0, \
            'Filter dimensions must be >= 0'

        if pool_param.stride is not None:
            self.stride_h = self.stride_w = pool_param.stride
        elif pool_param.stride_h is not None and \
                pool_param.stride_w is not None:
            self.stride_h = pool_param.stride_h
            self.stride_w = pool_param.stride_w
        else:
            raise Exception("Pooling Layer stride OR stride_h and stride_w"
                            "required.")

        if pool_param.pad is not None:
            self.pad_h = self.pad_w = pool_param.pad
        elif pool_param.pad_h is not None and \
                pool_param.pad_w is not None:
            self.pad_h = pool_param.pad_h
            self.pad_w = pool_param.pad_w
        else:
            raise Exception("Pooling Layer pad OR pad_h and pad_w"
                            "required.")

        assert self.pad_h < self.kernel_h and self.pad_w < self.kernel_w, \
            "Padding dimensions should be smaller than kernel dimensions"

    def get_top_shape(self, bottom):
        channels, height, width = bottom[0].shape
        pooled_height = (height + 2 * self.pad_h - self.kernel_h) \
            / self.stride_h + 1
        pooled_width = (width + 2 * self.pad_w - self.kernel_w) \
            / self.stride_w + 1
        return bottom.shape[:2] + (pooled_height, pooled_width)

    def setup(self, bottom, top):
        self.mask = Array.zeros_like(top)

    # @meta
    def forward(self, bottom, top):
        for n in range(bottom.shape[0]):
            for channel in range(bottom.shape[1]):
                max_pool(bottom[n, channel], top[n, channel],
                         self.mask[n, channel],
                         (self.kernel_h, self.kernel_w),
                         (self.pad_h, self.pad_w), (self.stride_h,
                         self.stride_w))

    def backward(self, bottom_diff, top_diff):
        for n in range(top_diff.shape[0]):
            for c in range(top_diff.shape[1]):
                max_pool_backward(top_diff[n, c], bottom_diff[n, c],
                                  self.mask[n, c])
