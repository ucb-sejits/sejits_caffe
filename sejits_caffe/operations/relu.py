from sejits_caffe.types.array import specialized_dispatch, smap2


def relu_factory(negative_slope):
    @smap2
    def relu(scale_element, pre_mask_element):
    	#pre_mask: need to check if pre_mask is positive to create a mask
        return scale_element*((pre_mask_element > 0) + negative_slope * (pre_mask_element <= 0))

    return relu


relu_cache = {}


@specialized_dispatch
def relu(scale, pre_mask, output, negative_slope):
    if negative_slope not in relu_cache:
        relu_cache[negative_slope] = \
            relu_factory(negative_slope)
    return relu_cache[negative_slope]

relu.num_args = 3
