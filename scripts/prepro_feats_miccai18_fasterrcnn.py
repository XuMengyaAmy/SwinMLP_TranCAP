"""
Preprocess a raw json dataset into features files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: two folders of features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import random
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from PIL import Image

def seed_everything(seed=123): 
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    seed(123) # make reproducible
    seed_everything(123) # full set to make reproducible

    dir= params['output_dir']+'_'+'fasterrcnn_res18_features'
    if not os.path.isdir(dir):
        os.mkdir(dir)


    for i,img in enumerate(imgs):
        # # load the feature from fasterrcnn 
        #### for miccai18 dataset
        feature_path = os.path.join(params['fasterrcnn_feat_root'], img['filepath'], img['filename'].split('.')[0]+'.npy') # filename is frame000.png
        print('processing', feature_path)

        if not os.path.exists(feature_path):
            feature = np.zeros((6, 512))
        else:
            feature = np.load(feature_path)
        print('feature shape', feature.shape)

        # write to pkl
        np.save(os.path.join(dir, str(img['imgid'])), feature)

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', params['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='./data/data_miccai18/miccai18_caption.json', help='input json file to process into hdf5') # required=True, 
    parser.add_argument('--output_dir', default='./data/data_miccai18/miccai18', help='output h5 file')
    parser.add_argument('--fasterrcnn_feat_root', default='/home/ren2/data2/mengya/mengya_code/main_ImageCaptioning.pytorch/data/data_miccai18/fasterrcnn_res18_features')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
