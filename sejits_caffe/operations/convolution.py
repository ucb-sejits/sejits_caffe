from sejits_caffe.types.array import specialize


@specialize
def convolution_2d(data, weights, output, pad_h, pad_w, stride_h, stride_w):
    for y, x in output.indices():
        for j, i in weights.indices():
            y_in = y * stride_h - pad_h + j
            x_in = x * stride_w - pad_w + i
            if 0 <= y_in < data.shape[0] and 0 <= x_in < data.shape[1]:
                output[y, x] += weights[j, i] * data[y_in, x_in]
