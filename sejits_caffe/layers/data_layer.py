from sejits_caffe.layers.base_layer import BaseLayer
from cstructures import Array
import numpy as np

# import os
# os.environ['LMDB_FORCE_CFFI'] = '1'

import lmdb
import sejits_caffe.caffe_pb2 as caffe_pb2
import random


class DataTransformer(object):
    def __init__(self, param, phase):
        self.param = param
        self.phase = phase
        if param.HasField("mean_file"):
            blob_proto = caffe_pb2.BlobProto()
            with open(param.mean_file, "rb") as f:
                blob_proto.ParseFromString(f.read())
                # FIXME: Assuming float32
                self.mean = Array.array(
                    blob_proto.data._values).astype(np.float32)
        else:
            raise NotImplementedError()

    def transform(self, datum):
        channels, datum_height, datum_width = datum.channels, datum.height, \
            datum.width

        crop_size = self.param.crop_size
        do_mirror = self.param.mirror
        # scale = self.param.scale
        height = datum_height
        width = datum_width

        if crop_size:
            height = crop_size
            width = crop_size
            if self.phase == "train":
                h_off = random.randrange(datum_height - crop_size + 1)
                w_off = random.randrange(datum_width - crop_size + 1)
            else:
                h_off = (datum_height - crop_size) / 2
                w_off = (datum_width - crop_size) / 2

        transformed_data = Array.zeros(
            np.prod((channels, height, width)), np.float32)

        data = Array.fromstring(
            datum.data, dtype=np.uint8).astype(
                np.float32).reshape(channels, datum_height, datum_width)
        transformed_data = data[..., h_off:h_off + height, w_off:w_off + width]
        if do_mirror:
            for c in range(channels):
                transformed_data[c] = np.fliplr(transformed_data[c])

        return transformed_data
        # for c in range(channels):
        #     for h in range(height):
        #         for w in range(width):
        #             data_index = (c * datum_height + h_off + h) *
        #                 datum_width + w_off + w
        #             if do_mirror:
        #                 top_index = (c * height + h) * width + \
        #                       (width - 1 - w)
        #             else:
        #                 top_index = (c * height + h) * width + w

        #             datum_element = data[data_index]

        #             if self.param.HasField("mean_file"):
        #                 transformed_data[top_index] = \
        #                     (datum_element - self.mean[data_index]) * scale
        #             else:
        #                 raise NotImplementedError()
        # return transformed_data.reshape((channels, height, width))


class DataLayer(BaseLayer):
    def get_top_shape(self):
        backend = self.layer_param.data_param.backend
        if backend == caffe_pb2.DataParameter.LMDB:
            self.db = lmdb.open(self.layer_param.data_param.source)
            txn = self.db.begin()
            self.cursor = txn.cursor().iternext()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(next(self.cursor)[1])
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

    def forward(self, top, top_label):
        datum = caffe_pb2.Datum()
        batch_size = self.layer_param.data_param.batch_size
        for i in range(batch_size):
            print("data layer batch: ", i)
            datum.ParseFromString(next(self.cursor)[1])
            top[i] = self.data_transformer.transform(datum)
            top_label[i] = datum.label
