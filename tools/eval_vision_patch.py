from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
# from captioning.data.dataloader import *
from captioning.data.dataloader_vision import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='log_best_models/miccai18/log_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='log_best_models/miccai18/log_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128/infos_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128-best.pkl',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                help='cpu or cuda')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

# log_best_models/miccai18/log_miccai18_Linear512_Vision_Swin_MLP_patchemd_separate_with_attmask_embdim128_Mar2/model-best.pth 
# log_best_models/miccai18/log_miccai18_Linear512_Vision_Swin_MLP_patchemd_separate_with_attmask_embdim128_Mar2/infos_miccai18_Linear512_Vision_Swin_MLP_patchemd_separate_with_attmask_embdim128_Mar2-best.pkl

# log_best_models/miccai18/log_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128/model-best.pth
# log_best_models/miccai18/log_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128/infos_miccai18_new_linear512_Vision_Swin_TransformerModel_patchemd_separate_with_attmask_embdim128-best.pkl


# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
print('========================')
print('batch_size', opt.batch_size)
print('========================')
vocab = infos['vocab'] # ix -> word mapping

pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

if opt.only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)): 
    # if results existed, then skip, unless force is on
    if not opt.force:
        try:
            if os.path.isfile(result_fn):
                print(result_fn)
                json.load(open(result_fn, 'r'))
                print('already evaluated')
                os._exit(0)
        except:
            pass

    predictions, n_predictions = torch.load(pred_fn)
    lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
    print(lang_stats)
    os._exit(0)

# At this point only_lang_eval if 0
if not opt.force:
    # Check out if 
    try:
        # if no pred exists, then continue
        tmp = torch.load(pred_fn)
        # if language_eval == 1, and no pred exists, then continue
        if opt.language_eval == 1:
            json.load(open(result_fn, 'r'))
        print('Result is already there')
        os._exit(0)
    except:
        pass

# Setup the model
opt.vocab = vocab
print('caption_model', opt.caption_model) # caption_model Linear512_Vision_Swin_MLP_patchemd_separate_with_attmask_embdim12
model = models.setup(opt)
del opt.vocab
# model.load_state_dict(torch.load(opt.model, map_location='cpu')) 
model.load_state_dict(torch.load(opt.model, map_location='cpu'), strict=False)# strict=False

model.to(opt.device)
model.eval()
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    print('111111111111111111111')
    loader = DataLoader(opt)
else:
    # print('2222222222222')
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
        vars(opt)) # author


print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))


# Evaluate on COCO-like test set
# $ python tools/eval.py --input_json cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0
# python tools/eval_vision_patch.py --input_json data/data_miccai18/miccai18_afterprepro.json --input_label_h5 none --language_eval 0
# can work: CUDA_VISIBLE_DEVICES=0 python tools/eval_vision_patch.py --input_json data/data_miccai18/miccai18_afterprepro.json --input_label_h5 none --language_eval 1 --batch_size 1
            
# CUDA_VISIBLE_DEVICES=0 python tools/eval_vision_patch.py --input_json data/data_miccai18/miccai18_val.json --input_label_h5 none --language_eval 1 --batch_size 1         
# 
            
            # # eval model
            # eval_kwargs = {'split': 'val',
            #                 'dataset': opt.input_json} # val data
            # eval_kwargs.update(vars(opt))
            # val_loss, predictions, lang_stats = eval_utils.eval_split(
            #     dp_model, lw_model.crit, loader, eval_kwargs)

            # # mengya: the adopted reduce_on_plateau: True. 
            # if opt.reduce_on_plateau:
            #     if 'CIDEr' in lang_stats:
            #         optimizer.scheduler_step(-lang_stats['CIDEr']) 
            #     else:
            #         optimizer.scheduler_step(val_loss)

            # # Save model if is improving on validation result
            # if opt.language_eval == 1:
            #     current_score = lang_stats['CIDEr']
            #     current_Bleu_4= lang_stats['Bleu_4']
            # else:
            #     current_score = - val_loss