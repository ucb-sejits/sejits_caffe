class BaseLayer(object):
    def __init__(self, param):
        self.layer_param = param
        # TODO:  Initialize with proto blob

    def forward(self, bottom, top):
        raise NotImplementedError()

    def backward(self, bottom, propagate_down, top):
        raise NotImplementedError()
