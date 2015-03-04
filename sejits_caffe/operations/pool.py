from sejits_caffe.types.array import specialize, specialized_dispatch



def max_pool_factory(padding, stride, kernel_size):
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size

    @specialize
    def max_pool(data, output, mask):
        for y, x in output.indices():
            y_start = max(y * stride_h - pad_h, 0)
            x_start = max(x * stride_w - pad_w, 0)
            y_end = min(y_start + kernel_h, data.shape[0])
            x_end = min(x_start + kernel_w, data.shape[1])
            # TODO: Add product or some other construct to make this neater
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    if data[yy, xx] > output[y, x]:
                        output[y, x] = data[yy, xx]
                        mask[y, x] = yy * data.shape[1] + xx

    return max_pool


pool_cache = {}


@specialized_dispatch
def max_pool(data, output, mask, kernel_size, padding=(0, 0), stride=(1, 1)):
    if (padding, stride, kernel_size) not in pool_cache:
        pool_cache[padding, stride, kernel_size] = \
            max_pool_factory(padding, stride, kernel_size)
    return pool_cache[padding, stride, kernel_size]

max_pool.num_args = 3
