from sejits_caffe.layers.base_layer import BaseLayer
from cstructures import Array
import numpy as np

import os
os.environ['LMDB_FORCE_CFFI'] = '1'

import lmdb
import sejits_caffe.caffe_pb2 as caffe_pb2
import itertools
import random


class DataTransformer(object):
    def __init__(self, param, phase):
        self.param = param
        self.phase = phase
        if param.HasField("mean_file"):
            blob_proto = caffe_pb2.BlobProto()
            with open(param.mean_file, "rb") as f:
                blob_proto.ParseFromString(f.read())
                shape = blob_proto.num, blob_proto.channels, \
                    blob_proto.width, blob_proto.height
                self.mean = Array.zeros(shape, np.float32)
                for i, j, k, l in itertools.product(shape):
                    index = i * np.prod(self.mean.shape[1:]) + j * \
                        np.prod(self.mean.shape[2:]) + k * \
                        np.prod(self.mean.shape[3:]) + l
                    self.mean[i, j, k, l] = blob_proto.data[index]
        else:
            raise NotImplementedError()

    def transform(self, datum):
        channels, datum_height, datum_width = datum.channels, datum.height, \
            datum.width

        crop_size = self.param.crop_size
        do_mirror = self.param.mirror
        scale = self.param.scale
        height = datum_height
        width = datum_width

        if crop_size:
            height = crop_size
            width = crop_size
            if self.phase == 'TRAIN':
                h_off = random.randrange(datum_height - crop_size + 1)
                w_off = random.randrange(datum_width - crop_size + 1)
            else:
                h_off = (datum_height - crop_size) / 2
                w_off = (datum_width - crop_size) / 2

        transformed_data = Array.zeros((channels, height, width), np.float32)

        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    data_index = (c * datum_height + h_off + h) * datum_width \
                        + w_off + w
                    if do_mirror:
                        top_index = (c * height + h) * width + (width - 1 - w)
                    else:
                        top_index = (c * height + h) * width + w

                    datum_element = datum.float_data(data_index)

                    if self.param.HasField("mean_file"):
                        transformed_data[top_index] = \
                            (datum_element - self.mean[data_index]) * scale
                    else:
                        raise NotImplementedError()


class DataLayer(BaseLayer):
    def get_top_shape(self):
        backend = self.layer_param.data_param.backend
        if backend == 'lmdb':
            self.db = lmdb.open(self.layer_param.data_param.source)
            with self.db.begin() as txn:
                cursor = txn.cursor()
                self.cursor = iter(cursor)
                datum = caffe_pb2.Datum()
                datum = datum.ParseFromString(next(self.cursor))
                crop_size = self.layer_param.transform_param.crop_size
                if crop_size > 0:
                    return self.layer_param.data_param.batch_size, \
                        datum.channels, crop_size, crop_size
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError(backend)

    def setup(self, top, top_label):
        self.data_transformer = \
            DataTransformer(self.layer_param.transform_param, self.phase)

    def forward(self, bottom, top, top_label):
        datum = caffe_pb2.Datum()
        batch_size = self.layer_param.data_param.batch_size
        for i in range(batch_size):
            datum = datum.ParseFromString(next(self.cursor))
            top[i] = self.data_transformer.transform(datum)
            top_label[i] = datum.label
