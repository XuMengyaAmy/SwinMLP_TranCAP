'''
This code is used to generate the daisi_test.json file.

The input json file has the following format:
images": [{"sentids": [2809], "imgid": 2809, 
"sentences": [{"tokens": ["vesicouterine", "pouch", "between", "urinary", "bladder", "and", "uterus"], 
"raw": "vesicouterine pouch between urinary bladder and uterus", "imgid": 2809, "sentid": 2809}],
"split": "train", "filename": "2809.jpg"},

Becuase each coco image has 5 captions. the image_id is not same as id. The id is caption_id

The coco-style output json file just contain the imgid which belong to split: test
the output json file has the following format:
info	{…}
images	[…]
licenses	[…]
type	"captions"
annotations	[…]
{
"images": [{"license": 3, "url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg", "file_name": "COCO_val2014_000000391895.jpg", "id": 391895, "width": 640, "date_captured": "2013-11-14 11:18:45", "height": 360}, 
           {"license": 4, "url": "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg", "file_name": "COCO_val2014_000000522418.jpg", "id": 522418, "width": 640, "date_captured": "2013-11-14 11:38:44", "height": 480},

"annotations": [{"image_id": 203564, "id": 37, "caption": "A bicycle replica with a clock as the front wheel."}, 
                {"image_id": 179765, "id": 38, "caption": "A black Honda motorcycle parked in front of a garage."}, {"image_id": 322141, "id": 49, "caption": "A room with blue walls and a white sink and door."}, 
                {"image_id": 16977, "id": 89, "caption": "A car that seems to be parked illegally behind a legally parked car"}, {"image_id": 106140, "id": 98, "caption": "A large passenger airplane flying through the air."}, 
}

=============
The real output json file for daisi dataset
{
    'images':
    "annotations":
}
'''

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
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image


def seed_everything(seed=123):
    print('=================== set the seed :', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(params):
    seed(123) # make reproducible
    seed_everything(123) # full set to make reproducible
    
    test_output = {}

    images = []
    images_element = {}

    annotations = []
    annotations_element = {}


    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images'] # imgs is that big list, each element in this list is a dic

    for img in imgs:
        if img["split"] == "test":
            print("test id", img["imgid"])

            images_element = {"file_name": str(img["imgid"])+'.jpg', "id":img["imgid"], "split": "test"}
            images.append(images_element)

            annotations_element = {"image_id": img["imgid"], 
            "id": img["imgid"], 
            "caption": img["sentences"][0]["raw"]}
            annotations.append(annotations_element)
    
    test_output = {"images": images, "type": "captions", "annotations":annotations}

    
    with open(params['output_json'], 'w') as f:
        json.dump(test_output, f)

    print('Done')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='data/data_daisi/daisi_caption.json', help='input json file') # required=True, 
    parser.add_argument('--output_json', default='data/data_daisi/daisi_test.json', help='output json file') 
    
    args = parser.parse_args()
    params = vars(args) 
    print('parsed input parameters:')
    main(params)


