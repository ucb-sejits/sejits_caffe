#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
import os
from google.protobuf import text_format
import caffe_pb2

from layers.conv_layer import ConvLayer
from layers.relu_layer import ReluLayer
from layers.data_layer import DataLayer
from layers.lrn_layer import LRNLayer
from layers.pooling_layer import PoolingLayer
from layers.inner_product_layer import InnerProductLayer
from layers.dropout_layer import DropoutLayer
from layers.accuracy_layer import AccuracyLayer
from layers.softmax_loss_layer import SoftMaxWithLossLayer
import numpy as np

from IPython import embed


from sejits_caffe.types import Array


class Net(object):
    layer_type_map = {
        "Data": DataLayer,
        "Convolution": ConvLayer,
        "ReLU": ReluLayer,
        "LRN": LRNLayer,
        "Pooling": PoolingLayer,
        "InnerProduct": InnerProductLayer,
        "Dropout": DropoutLayer,
        "Accuracy": AccuracyLayer,
        "SoftmaxWithLoss": SoftMaxWithLossLayer,
    }

    def __init__(self, param_file):
        # importing net param from .prototxt 
        self.param = caffe_pb2.NetParameter()
        param_string = open(param_file).read()
        text_format.Merge(param_string, self.param)
        self.layers = []
        self.blobs = {}
        for layer_param in self.param.layer:
            bottom = []
            top = []
            for blob in layer_param.bottom:
                if blob not in self.blobs:
                    self.blobs[blob] = Array.zeros((5, 4, 256, 256), np.float32)
                bottom.append(self.blobs[blob])
            for blob in layer_param.top:
                if blob not in self.blobs:
                    self.blobs[blob] = Array.zeros((5, 4, 256, 256), np.float32)
                top.append(self.blobs[blob])
            layer = self.layer_type_map[layer_param.type](layer_param)
            layer.set_up(np.array(bottom).view(Array), np.array(top).view(Array))
            self.layers.append(layer)
        print(self.layers)


    def FilterNet(self, param, param_filtered):
        pass

    def AppendTop(self, param, layer_id, top_id, available_blobs, blob_name_to_idx):
        pass

    def AppendBottom(self, param, layer_id, bottom_id, available_blobs, blob_name_to_idx):
        pass

    def Initialize(self, in_param):
        pass


def main(argv):
    if len(argv) != 2:
        raise Exception('Usage: model .prototxt file')
    else:
        n = Net(sys.argv[1])
    embed()

#     L1 = ConvLayer(n.param.layer[2])
#     L2 = ConvLayer(n.param.layer[6])


if __name__ == '__main__':
    import sys
    main(sys.argv)
