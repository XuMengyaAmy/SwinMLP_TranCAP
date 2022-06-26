from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import captioning.utils.opts as opts


import captioning.models as models

from captioning.data.dataloader_vision import *

import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import LossWrapper

import tqdm
import random
def seed_everything(seed=123): # 3407, 123
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    seed_everything()
    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    print('===================================')
    print('You are using model:', opt.caption_model)
    print('===================================')
    del opt.vocab

    # ========= calculate parameters and flops ========== #
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters) # number of params: 99113079
    # print('attributes:', dir(model.model.encoder))
    if hasattr(model.model.encoder, 'flops'): # SwinMLP_Encoder
        flops = model.model.encoder.flops()
        # logger.info(f"number of GFLOPs: {flops / 1e9}")
        print('number of GFLOPs of Encoder:', {flops / 1e9}) # number of GFLOPs of Encoder: {10.889017344}
    # =================================================== #

    # # Load pretrained weights:
    # # mengya: continue training from saved model at this path. Path must contain files saved by previous training process:
    # # 'infos.pkl'         : configuration;
    # # 'model.pth'         : weights
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt) # mengya: lw_model is for training 

    # ======================= original one ================= #
    # Wrap with dataparallel
    num_gpu = torch.cuda.device_count()
    print('=====================================')
    print('The total available GPU_number:', num_gpu)
    
    dp_model = torch.nn.DataParallel(model) # mengya: dp_model is for evaluation
    dp_model.vocab = getattr(model, 'vocab', None)  # nasty
    dp_lw_model = torch.nn.DataParallel(lw_model) # mengya: dp_lw_model is for training
    # ====================================================== #




    ##########################
    #  Build optimizer
    ##########################
    # ********************** adopted optimizer ******************** #
    if opt.noamopt:
        print(' ============ mengya: using utils.get_std_opt !!!!!!! =============')
        print('mengya: torch.optim.Adam optimizer and the learning rate updating strategy adopts transformer type.')
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer', 
        'vision_transformer', 
        'SwinMLP_TranCAP',
        'SwinMLP_TranCAP_L',
        'Swin_TranCAP',
        'Swin_TranCAP_L',
        'Video_Swin_TranCAP',
        'Video_Swin_TranCAP_L',
        'Video_SwinMLP_TranCAP_L'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    # ************************************************** #

    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    
    # mengya: load the optimizer is for resume training.
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict']) ######### about the dataloader
    
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None) # ============= enter this
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True

    # Assure in training mode
    dp_lw_model.train() # mengya: it is similar with model.train()
    best_epoch = 0

    # Start training
    # while True 不能直接改成 “for epoch in range(opt.max_epochs):“， 因为 while True用的loader.get_batch('train')，是以 batch来循环更新的
    # 而 “for epoch in range(opt.max_epochs):“ 是以 一整个数据集 做一次循环， 而不是以 batch 为循环
    # 如果直接改成“for epoch in range(opt.max_epochs):“，就变成了一个epoch只循环一个batch。
    while True: # mengya: 如果出现错误的话，可以继续循环。 while True 语句中一定要有结束该循环的break语句，否则会一直循环下去的。
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            print('==> Train finished.')
            print('Best epoch: ', best_epoch, ' Bleu_4 at best epoch: ', Bleu_4_at_best_epoch, ' best CIDEr: ', best_val_score)
            break
        # print('=================== current epoch:', epoch)

        # if epoch_done:
        #     sc_flag = False
        #     struc_flag = False
        #     drop_worst_flag = False
        #     epoch_done = False

        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    print('Assign the learning rate')
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                print('Assign the scheduled sampling prob')
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                print('start self critical training')
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False
            
            # If start structure loss training
            if opt.structure_after != -1 and epoch >= opt.structure_after:
                print('start structure loss training')
                struc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                struc_flag = False
            if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                drop_worst_flag = True
            else:
                drop_worst_flag = False

            epoch_done = False

            # # mengya: show the progress bar
            tq = tqdm.tqdm(total=loader.get_dataset_size('train')+1000) # mengya: len(loader.dataset) in this case = 113287 train + 5000 val + 5000 test = 123287
            tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
            # tq.set_description('Train: Epoch {}'.format(epoch)) 
            
        start = time.time()

        if opt.use_warmup and (iteration < opt.noamopt_warmup):
            opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup # opt.learning_rate：adopt 5e-4 mengya: warm up the learing rate 
            utils.set_lr(optimizer, opt.current_lr)

        # tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr'])) 

        # ***************** Load data. It is similar with  "for data in loader: "" *********************** #
        # Load data from train split (0)
        data = loader.get_batch('train') # mengya: the batch train data ############################
        # print('Read data:', time.time() - start) # my: Read data means the time used rather than the data percentage send into the model

        torch.cuda.synchronize() # 用于正确测试时间的代码
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        # print('labels', labels.shape) # labels torch.Size([50, 1, 18])
        # ************************************************************************************************* #
        
        optimizer.zero_grad()

        # mengya: output = model (input, labels)
        model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag)

        if not drop_worst_flag:
            loss = model_out['loss'].mean() # ============== mengya: loss = crit()
        else:
            loss = model_out['loss']
            loss = torch.topk(loss, k=int(loss.shape[0] * (1-opt.drop_worst_rate)), largest=False)[0].mean()

        loss.backward()

        if opt.grad_clip_value != 0:
            # grad_clip_value means clip gradients at this value/max_norm, 0 means no clipping
            # the adopted grad_clip_value: 0.1
            getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)

        optimizer.step()
        train_loss = loss.item()

        # tq.update()
        tq.update(opt.batch_size)
        
        torch.cuda.synchronize()
        end = time.time()

        # if struc_flag:
        #     print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
        #         .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
        # elif not sc_flag:
        #     print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
        #         .format(iteration, epoch, train_loss, end - start))
        # else:
        #     print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
        #         .format(iteration, epoch, model_out['reward'].mean(), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            # mengya, record the current epoch
            current_epoch = epoch
            # print('============ current_epoch:', current_epoch)
            epoch += 1 # set the start epoch for the resume training, rememr to add 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
            tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
            elif struc_flag:
                tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

            histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
            histories['lr_history'][iteration] = opt.current_lr
            histories['ss_prob_history'][iteration] = model.ss_prob

        # update infos
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['loader_state_dict'] = loader.state_dict()

        if epoch_done:
        
            tq.close()# *************************

            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json} # val data
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(
                dp_model, lw_model.crit, loader, eval_kwargs)

            # mengya: the adopted reduce_on_plateau: True. 
            if opt.reduce_on_plateau:
                if 'CIDEr' in lang_stats:
                    optimizer.scheduler_step(-lang_stats['CIDEr']) 
                else:
                    optimizer.scheduler_step(val_loss)

            # Write validation result into summary
            tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    tb_summary_writer.add_scalar(k, v, iteration)
            histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            
            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
                current_Bleu_4= lang_stats['Bleu_4']

                current_Bleu_4 = lang_stats['Bleu_4']
                current_METEOR = lang_stats['METEOR']
                current_ROUGE_L = lang_stats['ROUGE_L'] 
                current_SPICE = lang_stats['SPICE']
                current_WMD = lang_stats['WMD'] 
            else:
                current_score = - val_loss

            best_flag = False

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True

            # Dump miscalleous informations
            infos['best_val_score'] = best_val_score
            
            # utils.save_checkpoint(opt, model, infos, optimizer, histories)
            if opt.save_history_ckpt:
                utils.save_checkpoint(opt, model, infos, optimizer,
                    append=str(epoch) if opt.save_every_epoch else str(iteration))

            if best_flag:
                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                print('saving the best model for the whole training process')
                utils.save_checkpoint(opt, model, infos, optimizer, append='best')
                best_epoch = current_epoch # mengya: best epoch is the epoch which has the best CIDEr
                Bleu_4_at_best_epoch = current_Bleu_4

                METEOR_at_best_epoch = current_METEOR
                ROUGE_L_at_best_epoch = current_ROUGE_L
                SPICE_at_best_epoch = current_SPICE
                WMD_at_best_epoch = current_WMD
                # best_model = model

                if current_epoch < 50:
                    print('saving the best model for the first 50 epochs')
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best_within50epochs')

                if current_epoch < 100:
                    print('saving the best model for the first 100 epochs')
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best_within100epochs')

            print('======================================================================================================================')
            print('Until now: Best epoch: ', best_epoch, ' Bleu_4 at best epoch: ', Bleu_4_at_best_epoch, ' best CIDEr: ', best_val_score)
            print('METEOR: ', METEOR_at_best_epoch,  ' ROUGE_L: ', ROUGE_L_at_best_epoch, ' SPICE: ', SPICE_at_best_epoch, ' WMD: ', WMD_at_best_epoch)
            print('======================================================================================================================')


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed_everything()
    print('================== video vision parameters ==========================')
    print('batch_size        :', opt.batch_size)
    print('max_epochs        :', opt.max_epochs)
    print('patch_norm        :', opt.patch_norm)
    print('ape               :', opt.ape)

    print('patch_size        :', opt.patch_size)
    print('dim               :', opt.dim)
    print('rnn_size          :', opt.rnn_size)
    print('=====================================================================')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    train(opt)



# command to run: python tools/train.py --id fc --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
# python tools/train.py --cfg configs/fc.yml --id fc


# python tools/train_mengya_simple.py --cfg configs/miccai18/transformer/transformer_patch.yml --id miccai18_transformer_patch_lr5e-4