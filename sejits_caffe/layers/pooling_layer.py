from sejits_caffe.types import Array
from sejits_caffe.operations import max_pool, meta
from sejits_caffe.layers.base_layer import BaseLayer

import numpy as np


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

    def setup(self, top, bottom):
        self.mask = Array.empty_like(top)

    @meta
    def forward(self, bottom, top):
        for n in range(bottom.shape[0]):
            for channel in range(bottom.shape[1]):
                max_pool(bottom[n, channel], top[n, channel],
                         self.mask[n, channel],
                         (self.kernel_h, self.kernel_w),
                         (self.pad_h, self.pad_w), (self.stride_h,
                         self.stride_w))
