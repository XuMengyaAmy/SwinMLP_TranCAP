from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six

input_path = '/home/ren2/data2/mengya/mengya_code/main_ImageCaptioning.pytorch/data/data_daisi/daisi_512_att/35.npz'
# 9.npz

def load_npz(x):
    x = np.load(six.BytesIO(x))

input = open(os.path.join(input_path), 'rb').read()
print('input', input)
print('input shape:', input.shape)