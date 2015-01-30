#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
import os
from google.protobuf import text_format
import caffe_pb2

from layers.conv_layer import ConvLayer
import numpy as np

from IPython import embed
class net:
    def __init__(self, param_file):
        #importing net param from .prototxt 
        self.param = caffe_pb2.NetParameter()
        param_string = open(param_file).read()
        text_format.Merge(param_string, self.param)

    def FilterNet(self, param, param_filtered):
        pass

    def AppendTop(self, param, layer_id, top_id, available_blobs, blob_name_to_idx):
        pass

    def AppendBottom(self, param, layer_id, bottom_id, available_blobs, blob_name_to_idx):
        pass

    def Initialize(self, in_param):
        pass
        '''
        L1 = ConvLayer(in_param.layer[2])
        L2 = ConvLayer(in_param.layer[3])
        '''


def main(argv):
    '''
    if len(argv) != 2:
        print 'Usage: model .prototxt file'
    else:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(sys.argv[1]).read(), net)
    '''    
    
    n = net(sys.argv[1])
    embed()

    #B1 = np.zeros((100, 3, 225*225), np.floats32)
    #B2 = np.zer

    L1 = ConvLayer(n.param.layer[2])

    L2 = ConvLayer(n.param.layer[6])

    
'''
        print 'Drawing net to %s' % sys.argv[2]
        caffe.draw.draw_net_to_file(net, sys.argv[2])
'''

if __name__ == '__main__':
    import sys
    main(sys.argv)
