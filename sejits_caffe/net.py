#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
import os
from google.protobuf import text_format
import caffe_pb2

from IPython import embed
class net:
    def __init__(self, param_file):
        self.param = caffe_pb2.NetParameter()
        param_string = open(param_file).read()
        text_format.Merge(param_string, self.param)

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

    
'''
        print 'Drawing net to %s' % sys.argv[2]
        caffe.draw.draw_net_to_file(net, sys.argv[2])
'''

if __name__ == '__main__':
    import sys
    main(sys.argv)
