class BaseLayer(object):
    def forward(self, bottom, top):
        raise NotImplementedError()

    def backward(self, bottom, propagate_down, top):
        raise NotImplementedError()
