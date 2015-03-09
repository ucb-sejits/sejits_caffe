from sejits_caffe.layers.base_layer import BaseLayer


class LossLayer(BaseLayer):
    """docstring for LossLayer"""
    def __init__(self, param):
        super(LossLayer, self).__init__(param)
        if param.loss_weight_size == 0:
            self.param.loss_weight_size = 1
