from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle as pkl
from glob import glob
import numpy as np
import tensorflow as tf


class Processor(object):
    """
        Handles the reading and  preprocessing CIFAR-10

        NOTE:
            We assume that data_dir has the unzipped CIFAR-10 dataset
            downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, data_dir, batch_size=100):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def get_batch(self):
        """
            Loops through the dataset and keeps yielding image batches.
            
            It is possible that at the edges you get batches smaller than
            self.batch_size. Please file an issue if this is an issue.
        """
        files = glob(self.data_dir + "/data_batch_*")
        files.sort()
        assert len(files) == 5, "Expected 5 data_batch files, found {}".format(
            len(files))
        while (True):
            for f in files:
                tf.logging.info("Processing %s", f)
                data = pkl.load(open(f, "rb"), encoding='bytes')
                data = data[b"data"]
                data = data.reshape(-1, 3, 32, 32)
                data = data.transpose(0, 2, 3, 1)
                data = data.astype(np.float32)
                data = (data - 127.0) / 127.0  # Not 127, as (255 / 127) > 1
                assert np.min(data) >= -1, "min is {}".format(np.min(data))
                assert np.max(data) <= 255 / 127, "max is {}".format(
                    np.max(data))
                for i in range(0, len(data), self.batch_size):
                    yield data[i:i + self.batch_size]

            tf.logging.warn("Exhausted training data, looping back...")
