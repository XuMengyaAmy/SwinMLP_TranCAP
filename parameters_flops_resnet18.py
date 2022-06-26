# https://blog.csdn.net/weixin_44966641/article/details/120104600
# Transformer-FLOPs推导过程 https://its201.com/article/unamable/120731376
import torch
from torchvision.models import resnet18
# from torchvision.models import vit_b_16 # not available  # https://github.com/pytorch/vision/blob/289fce29b3e2392114aadbe7a419df0f2e3ac1be/torchvision/models/vision_transformer.py
from torchvision import models
from thop import profile
model = resnet18()
input = torch.randn(9, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print('flops:{}'.format(flops)) # / 1e9
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('GFLOPs:{}'.format(flops / 1e9)) # / 1e9
print('Params = ' + str(params/1000**2) + 'M')

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)
# input = torch.randn(1, 3, 224, 224)
# flops:1819066368.0
# FLOPs = 1.819066368G
# GFLOPs:1.819066368
# Params = 11.689512M
# number of params: 11689512


# input = torch.randn(9, 3, 224, 224)
# flops:16371597312.0
# FLOPs = 16.371597312G
# GFLOPs:16.371597312
# Params = 11.689512M
# number of params: 11689512

# Transformer-FLOPs推导过程: https://its201.com/article/unamable/120731376
# Input假设是一个维度为vocab的向量， 512 in captioning case
# output假设是一个 d_out
# Flop is related with batch_size and GPU number
# tgt_vocab = vocab

vocab = 512 # vocab可以调整
d_out = 1
tgt_vocab = 37

d_model=512 
d_ff=2048 
h=8 
d_k=64

encoder = 4 * vocab *d_model +vocab + 6 * (16* vocab *d_model + h * (6*d_model*vocab*d_k - 4*vocab*d_k + 4*d_k*vocab*vocab + 3*vocab*vocab - vocab) +2 * vocab * d_model *d_model - vocab * d_model + 2 * vocab + 4 * vocab * d_model *d_ff)
decoder = d_out * ( 4 * d_model +1) + 6 * (16 * d_out *d_model + 2 * d_out * d_model * d_model + 4*d_model*d_out*d_ff + 4 * d_model * vocab + vocab +2 * d_model * d_model * d_out + h * (8 * d_model * d_out * d_k - 6 * d_k + 4 * d_k * d_out * d_out + 3 * d_out * d_out - 2 * d_out +4 * d_model * vocab * d_k - 2 * d_k * vocab + 4 * d_k * d_out * vocab + 3 * vocab * d_out))
generator = 2 * tgt_vocab * d_out - d_out + 2 * d_model * tgt_vocab * d_out

GFLOPs_result = (encoder + decoder + generator) / 1e9
print('GFLOPs_result', GFLOPs_result) # GFLOPs_result 25.879521914

encoder_GFLOPs_result = encoder / 1e9
print('encoder GFLOPs_result', encoder_GFLOPs_result) # encoder GFLOPs_result 22.6046592

decoder_generator_GFLOPs_result = (decoder + generator) / 1e9
print('decoder + generator GFLOPs_result', decoder_generator_GFLOPs_result) # decoder + generator 3.274862714



