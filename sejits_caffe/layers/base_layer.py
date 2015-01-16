class BaseLayer(object):
    def forward(self, bottom, top):
        raise NotImplementedError()

    def backward(self, bottom, top):
        raise NotImplementedError()
