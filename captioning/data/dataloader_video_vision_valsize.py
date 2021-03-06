# ====================================== mengya: this dataloader is about video vision dataloader, sending miccai18 image sequentially ========================== #
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

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        # print('db_path: ', db_path)
        # db_path:  data/cocotalk_fc
        # db_path:  data/cocotalk_att
        # db_path:  data/cocotalk_box

        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))

        # ======== added by mengya ============== #
        elif self.ext == '.jpg':
            from PIL import Image
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
            # preprocess = transforms.Compose([
            #     transforms.Resize((224, 224))])
            def load_img(x):
                # print("input x:", x)
                img = Image.open(x).convert('RGB') # x is image path
                img = preprocess(img)
                # img = np.asarray(img, np.float32) / 255
                # img = np.array(img).transpose([2,0,1])
                return img

            self.loader = load_img
        # ======================================= #

        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        
        
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir' ################ my:enter this step
            
        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            if self.ext == '.jpg':
                # print('input img path:', os.path.join(self.db_path, key))
                # /home/ren2/data2/mengya/mengya_dataset/COCO2014/val2014/COCO_val2014_000000503135.jpg
                f_input = os.path.join(self.db_path, key)
            else:
                f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read() #################### my:enter this step

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input) # my:return the feature of one image

        return feat

# ================= mengya: for temporal process ===================== #
#                        seq4 ends with 567   seq6 start with 717
# seq_set_train =     [2,   3,   4,                     6,   7,   9,   10,  11,  12,  14, 15] # # train  1560
# num_each_seq_train [137, 134, 148,                   129, 149, 135, 149, 138, 146, 147, 148]
#                     149,  286, 420                   717, 846

# seq_set_val =     [1,   5,   16] # # val  447
# num_each_seq_val [149, 149, 149]

def removeCommonElements(a, b):
    for e in a[:]:
        if e in b:
            a.remove(e)
            # b.remove(e)
    return a 

def get_useful_start_idx(sequence_length, list_each_length, list_seq, val_seq):
    # list_seq = [1, 2, 3,  4, 5, 6,  7,  9,  10, 11, 12, 14, 15, 16]
    # list_each_length = [149, 137, 134, 148, 149, 129, 149, 135, 149, 138, 146, 147, 148, 149]
    list_seq_start_id = [0,]
    for i in range(1, len(list_seq)):
        # list_seq_start.append()
        temp_seq_start = np.sum(list_each_length[:i])
        list_seq_start_id.append(temp_seq_start)
    # print('list_seq_start_id ============',list_seq_start_id) # [0, 149, 286, 420, 568, 717, 846, 995, 1130, 1279, 1417, 1563, 1710, 1858]


    train_idx, val_idx = [], []
    dict_seqid_seqstartid = {m:n for m,n in zip(list_seq, list_seq_start_id)}
    dict_seqid_seqlen = {m:n for m,n in zip(list_seq, list_each_length)}
    # print(dict_each_len) # {1: 149, 2: 137, 3: 134, 4: 148, 5: 149, 6: 129, 7: 149, 9: 135, 10: 149, 11: 138, 12: 146, 14: 147, 15: 148, 16: 149}

    for i in val_seq:
        val_idx.extend( np.arange(dict_seqid_seqstartid[i], dict_seqid_seqstartid[i]+dict_seqid_seqlen[i]-(sequence_length-1)).tolist() )
    # print('val_idx', val_idx)

    train_seq = removeCommonElements(a = list_seq, b = val_seq)
    # print('train_seq', train_seq) # train_seq [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    for i in train_seq:
        train_idx.extend( np.arange(dict_seqid_seqstartid[i], dict_seqid_seqstartid[i]+dict_seqid_seqlen[i]-(sequence_length-1)).tolist() )
    # print('train_idx', train_idx)
    # train_useful_start_idx [149, 150, 151, ..., 279, 280, 281, || 286, , 287, 288, 289, 290,]s

    return train_idx, val_idx
# ==================================================================== #


class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)

        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json) # input_json: data/cocotalk.json
        self.info = json.load(open(self.opt.input_json)) # train split and val split together
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5) # input_label_h5: data/cocotalk_label.h5-
        print('image root: ', opt.input_img_dir)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:] ###################################
            self.label_end_ix = self.h5_label_file['label_end_ix'][:] ############################
        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)   ##########################
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory) ##########################
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory) ##########################

        # ======== added by mengya ============== #
        self.img_loader = HybridLoader(self.opt.input_img_dir, '.jpg', in_memory=self.data_in_memory) ######### added by mengya
        # ======================================== #

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # # =============================== for non-temporal case ========================= #
        # # separate out indexes for each of the provided splits
        # self.split_ix = {'train': [], 'val': [], 'test': []}
        # for ix in range(len(self.info['images'])): # ix??????0???????????????
        #     img = self.info['images'][ix]
        #     if not 'split' in img:
        #         self.split_ix['train'].append(ix) 
        #         self.split_ix['val'].append(ix)
        #         self.split_ix['test'].append(ix)

        #     elif img['split'] == 'train':
        #         self.split_ix['train'].append(ix) # ix ????????? ???id???????????????????????? self.split_ix['train']?????? ix ????????????0?????????????????????1??????????????? ????????????train+val????????????????????????
        #     elif img['split'] == 'val':
        #         self.split_ix['val'].append(ix)
        #     elif img['split'] == 'test':
        #         self.split_ix['test'].append(ix)

        #     elif opt.train_only == 0: # restval
        #         self.split_ix['train'].append(ix)

        # print('assigned %d images to split train' %len(self.split_ix['train']))
        # print('assigned %d images to split val' %len(self.split_ix['val']))
        # print('assigned %d images to split test' %len(self.split_ix['test']))
        # # ================================================================================= #

        # =============================== mengya: for temporal process ===================== #
        self.split_ix = {'train': [], 'val': [], 'test': []}

        train_useful_start_idx, val_useful_start_idx = get_useful_start_idx(sequence_length=5, list_each_length=[149, 137, 134, 148, 149, 129, 149, 135, 149, 138, 146, 147, 148, 149], \
             list_seq=[1, 2, 3,  4, 5, 6,  7,  9,  10, 11, 12, 14, 15, 16], val_seq=[1, 5, 16])
        np.random.shuffle(train_useful_start_idx) # ???????????? Super important step which helps to shuffle these video clips, # shuffle train video clips, similar with Shuffle=True
        # # train_useful_start_idx ?????????????????????????????????????????????????????????????????????id???
        # # train_useful_start_idx ???????????? ??????????????????????????????????????????????????????????????????????????????????????????????????????

        num_train_we_use = len(train_useful_start_idx)
        num_val_we_use = len(val_useful_start_idx)
        print('num of video clips to train = num_train_we_use =', num_train_we_use)
        print('num of video clips to val = num_val_we_use =', num_val_we_use)

        self.train_idx = []
        for i in range(num_train_we_use):
            for j in range(5): # self.opt.sequence_length
                # self.train_idx.append(train_useful_start_idx[i] + j) # train_idx_80 should be same with train_idx
                self.split_ix['train'].append(train_useful_start_idx[i] + j)
        print('number of all frames of train video clips = len(train_idx) =', len(self.split_ix['train']))
        
        
        self.val_idx = []
        for i in range(num_val_we_use):
            for j in range(5): # self.opt.sequence_length
                # self.val_idx.append(val_useful_start_idx[i] + j)
                self.split_ix['val'].append(val_useful_start_idx[i] + j)

        print('number of all frames of val video clips = len(val_idx) =', len(self.split_ix['val']))

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))
        # ================================================================================= #



    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    # https://blog.csdn.net/AWhiteDongDong/article/details/110233400
    # collate_fn?????????????????????__getitem__???????????????, collate_fn?????????????????????????????????????????????????????????????????????????????????
    def collate_func(self, batch, split): 
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []

        wrapped = False

        infos = []
        gts = []

        ix_batch = []
        
        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, \
                ix, it_pos_now, tmp_wrapped = sample

            ix_batch.append(ix)
     
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        
        fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True)) # why sort ???


        data = {} 
        data['fc_feats'] = np.stack(fc_batch)

        # ================== original one ============================ #
        # # merge att_feats
        # max_att_len = max([_.shape[0] for _ in att_batch]) 
        # data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32') # (batch_size, 196, 2048)
        # for i in range(len(att_batch)):
        #     data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        
        # data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        # for i in range(len(att_batch)):
        #     data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # # set att_masks to None if attention features have same length
        # if data['att_masks'].sum() == data['att_masks'].size:
        #     data['att_masks'] = None


        # ===================== added by mengya: image is the input ============== #
        # Here, att_feats is image actually, the shape is (3, 224, 224), original att_feats is (196, 2048)
        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch),att_batch[0].shape[0], att_batch[0].shape[1], att_batch[0].shape[2]], dtype = 'float32')
        for i in range(len(att_batch)):
            # data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i] # att_batch[i].shape[0] = 196, 
            data['att_feats'][i] = att_batch[i]
        data['att_masks'] = None
        # ======================================================================== #

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts # all ground truth captions of each images

        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}

        data['infos'] = infos

        data['ix'] = ix_batch

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor


        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        # index is elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        ix, it_pos_now, wrapped = index # why index can return three values, because index is from MySampler: elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        
        # # mengya: for temporal case
        # ix = index

        # print('ix inside getitem', ix)

        # ix ????????? ???id???????????????????????? self.split_ix['train']?????? ix ????????????0?????????????????????1??????????????? ????????????train+val???????????????????????? 
        # ??? ???train+val?????????????????????????????? self.info['images'] ??????????????? ix ??? self.info['images'][ix]['id']???????????????


        # # =============== original one =============== #
        # if self.use_att:
        #     att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))  # pass the key  {"split": "train", "file_path": "train2014/COCO_train2014_000000085322.jpg", "id": 85322}
        #     # Reshape to K x C
        #     att_feat = att_feat.reshape(-1, att_feat.shape[-1]) # after reshape operation: (14, 14, 2048) --> (196, 2048)

        #     if self.norm_att_feat:
        #         att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        #     if self.use_box:
        #         box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
        #         # devided by image width and height
        #         x1,y1,x2,y2 = np.hsplit(box_feat, 4)
        #         h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
        #         box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
        #         if self.norm_box_feat:
        #             box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        #         att_feat = np.hstack([att_feat, box_feat])
        #         # sort the features by the size of boxes
        #         att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        # else:
        #     att_feat = np.zeros((0,0), dtype='float32') ################## my: if we don't use att feature, the att_feat will be all 0

        
        # ======== added by mengya: image is the input ============== #
        if self.use_att:
            # print('img file:', str(self.info['images'][ix]['file_path'])) # {"split": "train", "file_path": "train2014/COCO_train2014_000000085322.jpg", "id": 85322}
            # val2014/COCO_val2014_000000503135.jpg
            img =  self.img_loader.get(str(self.info['images'][ix]['file_path'])) # "file_path": "seq_1/left_frames/frame041.png"
            # print('============= mengya ==================')
            # print('img shape:', img.shape) # img shape: (3, 224, 224)     
            att_feat = img
        else:
            att_feat = np.zeros((0,0), dtype='float32')
        # ======================================== #


        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32') ################## my: if we don't use fc feature, the fc_feat will be all 0
        
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None

        return (fc_feat,
                att_feat, seq,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])  # 1560+447 in non temporal case # make it perfect



class DataLoader:
    def __init__(self, opt):
        print('=================================')
        print('DataLoader for video vision with different valsize')
        print('=================================')

        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.val_batch_size = self.opt.val_batch_size
        self.dataset = Dataset(opt)


        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                # # sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
    
                # # # mengya; for temporal case
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=True) # shuffle must be false in train to keep the certain sequence in temporal case
                self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                        batch_size=self.batch_size,
                                        sampler=sampler,
                                        pin_memory=True,
                                        num_workers=4, # 4 is usually enough
                                        collate_fn=partial(self.dataset.collate_func, split=split),
                                        drop_last=False) 




            else:
                # sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
                # # mengya; for temporal case
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
                self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                        batch_size=self.val_batch_size,
                                        sampler=sampler,
                                        pin_memory=True,
                                        num_workers=4, # 4 is usually enough
                                        collate_fn=partial(self.dataset.collate_func, split=split),
                                        drop_last=False) 


            # self.loaders[split] = data.DataLoader(dataset=self.dataset,
            #                                       batch_size=self.batch_size,
            #                                       sampler=sampler,
            #                                       pin_memory=True,
            #                                       num_workers=4, # 4 is usually enough
            #                                       collate_fn=partial(self.dataset.collate_func, split=split),
            #                                       drop_last=False) # because of 2175/50=43.5 43*50-2175 = 2150-2175= -25, I set drop_last=True

                                                  # 447 val images
                                                  # 435 val images
                                                  # val_batchsize = 5

            self.iters[split] = iter(self.loaders[split])

    # added by mengya
    def get_dataset_size(self, split):
        return len(self.dataset.split_ix[split])
    
    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


# To understand it: https://www.cnblogs.com/marsggbo/p/11308889.html
class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self
    
    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        # print('!!!!!!!!!!!!!!!!!!!!!!!', self.iter_counter)
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        # print('elem', elem)
        
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)  ############ actual length

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
# 1560 non temporal
# 7580  temporal 
# num of video clips to train = num_train_we_use = 1516

# ******************************************************************************************************************#
    # # For back compatibility
    # if 'iterators' in infos:
    #     infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    # loader.load_state_dict(infos['loader_state_dict']) ######### about the dataloader

'''
        # train_loader_80 = DataLoader(
        #     train_dataset_80,
        #     batch_size=train_batch_size,
        #     sampler=SeqSampler(train_dataset_80, train_idx_80),
        #     num_workers=workers,
        #     pin_memory=False
        # )


        for i, data in enumerate(train_loader_80):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224) # ????????????batch size?????? sequence_length?????????(-1, sequence_length)

# # ============================================================================================== #

        # # ================= mengya: for temporal process ===================== #
        # # # step 1: get the num_each_seq_train, num_each_seq_val
        # # # step 2: we need match these id
        # train_useful_start_idx, val_useful_start_idx = get_useful_start_idx(sequence_length=5, list_each_length=[149, 137, 134, 148, 149, 129, 149, 135, 149, 138, 146, 147, 148, 149], \
        #      list_seq=[1, 2, 3,  4, 5, 6,  7,  9,  10, 11, 12, 14, 15, 16], val_seq=[1, 5, 16])
        
        # np.random.shuffle(train_useful_start_idx) # ???????????? Super important step which helps to shuffle these video clips, # shuffle train video clips, similar with Shuffle=True
        # # # train_useful_start_idx ?????????????????????????????????????????????????????????????????????id???
        # # # train_useful_start_idx ???????????? ??????????????????????????????????????????????????????????????????????????????????????????????????????
        
        # num_train_we_use = len(train_useful_start_idx)
        # num_val_we_use = len(val_useful_start_idx)
        # print('num of video clips to train = num_train_we_use =', num_train_we_use)
        # print('num of video clips to val = num_val_we_use =', num_val_we_use)

        # train_idx = []
        # for i in range(num_train_we_use):
        #     for j in range(5): # self.opt.sequence_length
        #         train_idx.append(train_useful_start_idx[i] + j) # train_idx_80 should be same with train_idx
        # print('number of all frames of train video clips = len(train_idx) =', len(train_idx))
        

        # val_idx = []
        # for i in range(num_val_we_use):
        #     for j in range(5): # self.opt.sequence_length
        #         val_idx.append(val_useful_start_idx[i] + j)
        # print('number of all frames of val video clips = len(val_idx) =', len(val_idx))
        # # ==================================================================== #
'''