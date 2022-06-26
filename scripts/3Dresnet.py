# Reference code: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/540a0ea1abaee379fa3651d4d5afbd2d667a1f49/datasets/videodataset.py#L38
# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
# https://github.com/pytorch/vision/tree/main/torchvision/models
# https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py
# https://github.com/xiadingZ/video-caption.pytorch/blob/master/prepro_feats.py
# tutorial to use torchvision video models: https://blog.csdn.net/qq_36627158/article/details/113791050

import torchvision.models.video as v_model
import torchvision.transforms as transform
import torchvision.transforms._transforms_video as v_transform
import torchvision.io as io
import torch
# from pretrainedmodels import utils
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import argparse
import random
from random import shuffle, seed
import string

from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from PIL import Image

from torchvision import transforms as trn

trn = trn.Compose([
                trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) # 图像分类网络输入大小是224*224

def seed_everything(seed=123): # original one is 123, 3407
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class my3DResnet(nn.Module):
    def __init__(self, resnet3d):
        super(my3DResnet, self).__init__()
        self.resnet3d = resnet3d
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, img, att_size=14):
        x = img.unsqueeze(0) # torch.Size([1, 3, 2, 224, 224]) to increase the dimensionality for batch_size
        print('x.shape', x.shape)
        x = self.resnet3d.stem(x)
        x = self.resnet3d.layer1(x)
        x = self.resnet3d.layer2(x)
        x = self.resnet3d.layer3(x)
        x = self.resnet3d.layer4(x) # x2.shape torch.Size([1, 512, 1, 14, 14])
        print('x2.shape', x.shape) 

        fc =  x.mean(4).mean(3).squeeze()
        att = F.adaptive_avg_pool3d(x,[1, att_size, att_size]).squeeze().permute(1, 2, 0) #  [1, att_size, att_size] D x H x W att 

        return fc, att

def clip_list(src_list,count):  #src_list为原列表，count为等分长度
    clip_back=[]
    if len(src_list) > count:
        for i in range(int(len(src_list) / count)):
            clip_a = src_list[count * i:count * (i + 1)]
            clip_back.append(clip_a)
        # last 剩下的单独为一组
        last = src_list[int(len(src_list) / count) * count:]
        if last:
            clip_back.append(last)
    else:  #如果切分长度不小于原列表长度，那么直接返回原列表
        clip_back = src_list  # 如果返回结构保持一致可以返回clip_back = [src_list]

    return clip_back



def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx

# def get_not_useful_start_idx(sequence_length, list_each_length):

trans = transform.Compose([
    v_transform.ToTensorVideo(),
    v_transform.RandomResizedCropVideo(224),
])
def extract_feats(params, my_3dresnet, video_path_list, sequence_length):
    C, H, W = 3, 224, 224
    # global C, H, W

    dir_fc = params['output_dir']+'_fc'
    dir_att = params['output_dir']+'_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    my_3dresnet.eval()
    for video_id in range(len(video_path_list)):
        imgid = video_id
        video_clip = torch.zeros((sequence_length, 3, H, W))# clip = torch.randn(3, 2, 224, 224)
        # print('video_clip shape', video_clip.shape) # torch.Size([2, 3, 224, 224])
        image_path_list = video_path_list[video_id]
        print('image_path_list', image_path_list)
        for i in range(len(image_path_list)):
            img = Image.open(image_path_list[i]).convert('RGB')
            img = img.resize((H, W)) # can't assign a Image to a torch.FloatTensor
            # print('img', type(img)) # img <class 'PIL.Image.Image'>
            img = trn(img)
            video_clip[i] = img
        # video_clip = trans(video_clip)
        # video_clip = video_clip.cuda()
        
        with torch.no_grad():
            # # clip = torch.randn(3, 2, 224, 224) correct
            video_clip = video_clip.permute(1,0,2,3) # [2, 3, 224, 224]-->[3, 2, 224, 224]
            # print('video_clip', video_clip.shape) 
            fc, att = my_3dresnet(video_clip.cuda())
        print('========================')
        print('fc', fc.shape) # tmp_fc shape torch.Size([2048]) --> torch.Size([512])
        print('att', att.shape) # tmp_att torch.Size([14, 14, 2048]) --> torch.Size([14, 14, 512])
        # write to pkl
        print('imgid', imgid)
        np.save(os.path.join(dir_fc, str(imgid)), fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, str(imgid)), feat=att.data.cpu().float().numpy())
    
def main(params):
    seed_everything()
    model = v_model.r3d_18(pretrained=True)
    # clip = torch.randn(3, 2, 224, 224)
    my_3dresnet = my3DResnet(model)
    # fc, att = my_3dresnet(clip)
    # print('fc', fc.shape) 
    # print('att', att.shape)
    my_3dresnet = my_3dresnet.cuda()
    sequence_length = 2

    # =================== prepare video path list ========================= #
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs) # num of total images
    print('N', N) # N 2007
    dataset_dic = {
        'seq_1': [], 'seq_2': [], 'seq_3': [], 'seq_4': [], 'seq_5': [], 'seq_6': [],
        'seq_7': [], 'seq_9': [], 'seq_10': [], 'seq_11': [], 'seq_12': [], 'seq_14': [],
        'seq_15':[], 'seq_16': []
                }
    for i,img in enumerate(imgs):
        img_path = os.path.join(params['images_root'], img['filepath'], 'left_frames', img['filename'])
        dataset_dic[img['filepath']].append(img_path)
        
    print('dataset_dic', dataset_dic)

    total_image = 0
    list_each_length = []
    all_images = []
    for name in dataset_dic.keys():
        list_each_length.append(len(dataset_dic[name]))
        total_image += len(dataset_dic[name])
        all_images.extend(dataset_dic[name])
    print('total_image', total_image) # m 2007
    print('list_each_length', list_each_length)
    print('all_images', len(all_images))

    
    # list_each_length [149, 137, 134, 148, 149, 129, 149, 135, 149, 138, 146, 147, 148, 149]
    useful_start_idx = get_useful_start_idx(sequence_length, list_each_length)
    # print('useful_start_idx', useful_start_idx)
    num_video_we_use = len(useful_start_idx)
    print('num_video_we_use', num_video_we_use) # num_video_we_use 1993

    used_idx = []
    for i in range(num_video_we_use):
        for j in range(sequence_length):
            used_idx.append(useful_start_idx[i] + j) # 0+0=0, 0+1=1; 147+0=147, 147+1=148,
            # to repeat the idx which are not used as the start idx multiple times
            
    # print('used_idx', used_idx)
    
    video_list=clip_list(used_idx,sequence_length)
    # print('video_list', video_list) # video list store the image id
    print('video_list', len(video_list)) # video_list 1993
    # conver the image id into the image path

    # ========= to repeat some image idx ============ #
    # list_each_length [149, 137, 134, 148, 149, 129, 149, 135, 149, 138, 146, 147, 148, 149]
    len_total = 0
    for id, length in enumerate(list_each_length):
        len_total+=length
        for num in range(sequence_length-1):
            # for nn in sequence_length
            add_seq = []
            for _ in range(sequence_length):
                add_seq.append(num+len_total-length)
            video_list.insert(len_total-length+num, add_seq)
    # print('==> video_list_after', video_list)
    print('==> video_list_after', len(video_list))
    # ============================================== #

    video_path_list = []
    for video_idx in range(len(video_list)):
        # print('video_idx', video_idx)
        images_path_list = []
        for image_idx in video_list[video_idx]:
            # print('image_idx', image_idx)
            images_path_list.append(all_images[image_idx])
        video_path_list.append(images_path_list)
    # print('video_path_list', video_path_list)
    print('length of video_path_list', len(video_path_list)) # length of video_path_list 1993
    # ============================================================ #   
    extract_feats(params, my_3dresnet, video_path_list, sequence_length)
    print('Done!')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='./data/data_miccai18/miccai18_caption.json', help='input json file to process into hdf5') # required=True, 
    parser.add_argument('--output_dir', default='./data/data_miccai18/miccai18_3dresnet18', help='output h5 file')
    # options
    parser.add_argument('--images_root', default='/home/ren2/data2/mengya/mengya_dataset/instruments18_caption', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    # parser.add_argument('--model', default='resnet18', type=str, help='resnet101, resnet152')
    # parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)