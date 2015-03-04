from sejits_caffe.types.array import specialized_dispatch, smap


def relu_factory(negative_slope):
    @smap
    def relu(elt):
        return max(elt, 0) + negative_slope * min(elt, 0)

    return relu


relu_cache = {}


@specialized_dispatch
def relu(data, output, negative_slope):
    if negative_slope not in relu_cache:
        relu_cache[negative_slope] = \
            relu_factory(negative_slope)
    return relu_cache[negative_slope]

relu.num_args = 2
