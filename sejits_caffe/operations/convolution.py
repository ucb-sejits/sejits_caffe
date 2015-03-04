from sejits_caffe.types.array import specialize, specialized_dispatch, Array
import numpy as np


def convolution_factory(padding, stride):
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    @specialize
    def convolution_2d(data, weights, output):
        for y, x in output.indices(cache_block=True):
            for j, i in weights.indices(unroll=True):
                y_in = y * stride_h - pad_h + j
                x_in = x * stride_w - pad_w + i
                if 0 <= y_in < data.shape[0] and 0 <= x_in < data.shape[1]:
                    output[y, x] += weights[j, i] * data[y_in, x_in]
    return convolution_2d


convolution_cache = {}


@specialized_dispatch(num_args=3)
def convolve(data, weights, output, padding=(0, 0), stride=(1, 1)):
    """
    Uses convolution factory generate a specialized convolution function
    for a specific padding and stride.  Variants are stored in convolution
    cache
    """
    if (padding, stride) not in convolution_cache:
        convolution_cache[(padding, stride)] = \
            convolution_factory(padding, stride)
    return convolution_cache[(padding, stride)]
