# ****************************** mengya: patch Transformer **************************** #
# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

# original one 
# class EncoderDecoder(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
        
#     def forward(self, src, tgt, src_mask, tgt_mask):
#         "Take in and process masked src and target sequences."
#         return self.decode(self.encode(src, src_mask), src_mask,
#                             tgt, tgt_mask)
    
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
    
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) # very first src_mask is used in decoder


# self.encode is different with self.encoder

# 50, 196, 2048

# 224 , patch_size = 16, 224/16= 14
# (50, 196, 2048)

# patch_size = 4 in Swin,  224/4 * 224/4 = 56 * 56 = 3136
#(50, 3136, 96)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src):
        return self.encoder(self.src_embed(src))
    
    def decode(self, memory, src_mask, tgt, tgt_mask): # # very first src_mask is used in decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# ================================================ Encoder (start) ========================================================================== #
# Encoder
# The encoder is composed of a stack of N=6 identical layers
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# We employ a residual connection around each of the two sub-layers, followed by layer normalization
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)).
# where Sublayer(x) is the function implemented by the sub-layer itself. We apply dropout (cite) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model=512.
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x))) # ====================== Norm, then Feed Forward, then Add 

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # original one
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # # for other token mixer: pooling
        # x = self.sublayer[0](x, lambda x: self.self_attn(x))

        return self.sublayer[1](x, self.feed_forward)
# ================================================ Encoder (End) ========================================================================== #


# ================================================ Decoder (Start) ======================================================================== #
# Decoder
# The decoder is also composed of a stack of N = 6  identical layers
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # print('src_mask in decoder', src_mask.shape) # src_mask in decoder torch.Size([5, 1, 256])
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. 
# Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #original one
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # correct self-attention in decoder
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # cross_attention, cross_mask

        # # for other token_mixer: pooling
        # x = self.sublayer[0](x, lambda x: self.self_attn(x))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x))


        return self.sublayer[2](x, self.feed_forward)

# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, 
# ensures that the predictions for position i can depend only on the known outputs at positions less than i.
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# ================================================ Decoder (End) ========================================================================= #


# ================================================ Multi-Head Self-Attention (MSA) (Start) ===================================================================== #
# core part of transformer model

def attention(query, key, value, mask=None, dropout=None): # Confrim shape of query, key, value, mask of this model and original transformer
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print('mask shape', mask.shape) # # mask shape torch.Size([5, 1, 1, 256])
        # print('scores', scores.shape) # # scores torch.Size([5, 8, 17, 3])

        # self-attention
        # mask shape torch.Size([5, 1, 17, 17])
        # scores torch.Size([5, 8, 17, 17])


        # Cross-attention in decoder
        # mask shape torch.Size([5, 1, 1, 256])  --> mask shape torch.Size([5, 1, 17, 3]) modify the mask
        # scores torch.Size([5, 8, 17, 3])  --> upsample it

        # RuntimeError: The size of tensor a (256) must match the size of tensor b (3) at non-singleton dimension 3

        # the example code to explain the mask: https://zhuanlan.zhihu.com/p/151783950
        scores = scores.masked_fill(mask == 0, float('-inf')) ################ mask is used here. We want to know in which positions, mask are 0 
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# How to call MultiHeadedAttention? Use self.self_attn(x, x, x, mask)
# we employ h=8 parallel attention layers, or heads.

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # 512 // 8 = 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # # src_mask in decoder torch.Size([5, 1, 256]) -->
            # print('mask in MultiHeadedAttention', mask.shape) # mask in MultiHeadedAttention torch.Size([5, 1, 1, 256]), mask in MultiHeadedAttention torch.Size([5, 1, 17, 17])
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # RuntimeError: shape '[3, -1, 8, 64]' is invalid for input of size 50176
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# ================================================ Multi-Head Self-Attention (MSA) (End) ===================================================================== #


# =============================================== Change MSA into Poiling (Start) ================================================ #
# replace the attention (e,g. core part of Transformer model) with others, what will happen?
# Pooling. Implementation is from https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
# ConvNext. Implementation is from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# swin_transformer WindowAttention: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
"""
Compared with transformer block, it replaces attention with an extremely simple non-parametric operator, pooling, to conduct only basic token mixing.
"""
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x
# =============================================== Change MSA into Poiling (End) ================================================ #


# # =============================================== Change MSA into W-MSA / SW-MSA (Start) ================================================ #
# #############################################
# # Some utilis
# #############################################
# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops

# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

#         x = x.view(B, H, W, C)

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"

#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# # The input image is divided into serveral windows.
# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     # print('========================== inside window_partition ===================')
#     # print('x', x.shape) # x torch.Size([1, 28, 28, 1])
#     B, H, W, C = x.shape  # 50, 56, 56, 96
#     # print('x inside window partition', x.shape)
#     # x inside window partition torch.Size([1, 28, 28, 1])
#     # x inside window partition torch.Size([1, 14, 14, 1]) 
#                                                                                    # 0   1   2 3   4  5
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # (50, 8, 7, 8, 7, 96)      shape '[1, 1, 16, 1, 16, 1]' is invalid for input of size 784
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # (50, 8, 8, 7, 7, 96) --> (3200, 7, 7, 96)--> (64*50, 7, 7, 96)
#     return windows


# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image
#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# #############################################
# # Core architecture of Swin Transformer model
# #############################################

# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         # print('x7 inside WindowAttention', x.shape) # x5 inside WindowAttention torch.Size([100, 196, 2048]) window_size: 14; torch.Size([320, 49, 96])

#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # shape '[100, 196, 3, 3, 682]' is invalid for input of size 120422400
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, input_resolution, num_heads, d_model, d_ff, dropout,
#                 window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio

#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)

#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

#         # mengya： It supports both of shifted and non-shifted window. e.g. W-MSA and SW-MSA
#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             # print('============== mengya !!!!!!!!')
#             # print('H, W', H, W) # H, W 28 28  224/8=28, thus the H and W of each patch is 28.; 56 56,  28 28,  14 14

#             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1  # 1, 56, 56, 1
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1  # (64*50, 7, 7, 1) windows: (num_windows*B, window_size, window_size, 1)
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [64*50, 49]
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [64*50, 1, 49] - [64*50, 49, 1] = [48, 48]
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask) # Buffers can be accessed as attributes using given names.

#     def forward(self, x):
#         # input_resolution=(patches_resolution[0] // (2 ** i_layer),
#         #                     patches_resolution[1] // (2 ** i_layer)),

#         H, W = self.input_resolution #  H, W = 28, 28 when patch_size = 8, 224/8=28. Here, input resolution in terms of "patch". 28 patches by 28 patches
#         B, L, C = x.shape # Here, the C is dim

#         # print('============= inside SwinTransformerBlock ==================')
#         # print('x3.shape', x.shape) # torch.Size([25, 784, 2048]) ;  torch.Size([5, 3136, 96])

#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)  # mengya: the first LN is applied here.
#         x = x.view(B, H, W, C)

#         # print('x4.shape', x.shape) # x4.shape torch.Size([25, 28, 28, 2048]); torch.Size([5, 56, 56, 96])

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         # print('x_windows2.shape', x_windows.shape) # x_windows.shape torch.Size([100, 14, 14, 2048]) ; torch.Size([320, 7, 7, 96])
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#         # print('x_windows2.shape', x_windows.shape) # x_windows.shape torch.Size([100, 196, 2048]) ; torch.Size([320, 49, 96])

#         # mengya: compute self attention between patches within that window, and we ignore the rest of the patches.
#         # mengya: the att_mask/mask here is self.attn_mask
#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C  

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)

#         # FFN
#         # mengya: FFN: Feed-Forward Network
#         x = shortcut + self.drop_path(x)  # the first residual connection

#         # original swin transformer model
#         x = x + self.drop_path(self.mlp(self.norm2(x)))  # MLP is applied here for classification case? but we need to change it for our captioning case

#         # # mengya: modify it for captioning case
#         # x = x + self.drop_path(self.feed_forward(self.norm2(x)))

#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self): # compute the flops for the model
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                 d_model, d_ff, dropout,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads,
#                                  d_model= d_model, d_ff=d_ff, dropout=dropout,
#                                  window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#     # mengya: BasicLayer includes 2 SwinTransformerBlock + Patch Merging
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x) # mengya: downsample is actually Patch Merging layer.
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops


# # SwinTransformerBlock --> Basiclayer (SwinTransformerBlock + patchmerging +) --> SwinTransformer_Encoder
# class SwinTransformer_Encoder(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """

#     def __init__(self, d_model, d_ff, dropout, 
#                  img_size=224, patch_size=4, in_chans=3,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, **kwargs): # original one are:  ndepths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]
#         super().__init__()

#         self.d_model = d_model
#         # self.num_classes = num_classes
#         self.num_layers = len(depths) # 一个虚线框代表一个layer, self.num_layers = 4
#         print('SwinTransformer_Encoder is composed of ', self.num_layers, ' layers' )
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 除了第一个虚线框，之后每经过一个虚线框，都乘2
#         self.mlp_ratio = mlp_ratio

#         norm_layer = nn.LayerNorm

#         # #split image into non-overlapping patches
#         # norm_layer = nn.LayerNorm
#         # self.patch_embed = PatchEmbed(
#         #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#         #     norm_layer=norm_layer if self.patch_norm else None)
        
#         # num_patches = self.patch_embed.num_patches
#         # patches_resolution = self.patch_embed.patches_resolution
#         # self.patches_resolution = patches_resolution

#         # # absolute position embedding
#         # if self.ape:
#         #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         #     trunc_normal_(self.absolute_pos_embed, std=.02)

#         # self.pos_drop = nn.Dropout(p=drop_rate)

#         # Replaced with mengya
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.patches_resolution = patches_resolution

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), # dim is C.   C, 2C, 4C, 8C. Here, the 2 cannot be changed.
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                                  patches_resolution[1] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                d_model=d_model, d_ff=d_ff, dropout=dropout,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint) # 4个虚线框(4个BasicLayer)中，有3个BasicLayer需要 “downsample=PatchMerging“
#             self.layers.append(layer)

#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)

#         # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}

#     def forward_features(self, x):
#         # # print('x inside SwinTransformer encoder', x.shape) # torch.Size([25, 3, 224, 224]) ; torch.Size([5, 3, 224, 224])
#         # x = self.patch_embed(x)
#         # if self.ape:
#         #     x = x + self.absolute_pos_embed
#         # x = self.pos_drop(x)
        
#         # print('x2 inside SwinTransformer encoder', x.shape) # torch.Size([25, 784, 2048])  （224/8） * (224/8) = 784; torch.Size([5, 3136, 96])

#         for layer in self.layers:
#             x = layer(x)
            
#         x = self.norm(x)  # B L C # 
#         # print('x outside layerblock', x.shape) # x outside layerblock torch.Size([5, 49, 768])

#         # x = self.avgpool(x.transpose(1, 2))  # B C 1
#         # x = torch.flatten(x, 1)
#         return x

#     def forward(self, x):
#         x = self.forward_features(x)

#         # print('*******************')
#         # print('x1 In SwinTransformer_Encoder', x.shape) # x1 In SwinTransformer_Encoder torch.Size([3, 49, 768])
#         # x = self.head(x)

#         # Added by menya for decoder
#         x = x.view(x.shape[0], -1, self.d_model) 
#         # print('x2 In SwinTransformer_Encoder', x.shape) # x2 In SwinTransformer_Encoder torch.Size([5, 147, 256])  # x2 In SwinTransformer_Encoder torch.Size([5, 3, 256])
        
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
# # =============================================== Change MSA into W-MSA / SW-MSA (End) ================================================ #

# ========= start =========
# mengya: cross attention in decoder
# mask: torch.Size([3, 1, 1, 196])
# scores torch.Size([3, 8, 17, 196])
# =========` end =========

# Cross-attention in decoder
# mask shape torch.Size([5, 1, 1, 256])  --> mask shape torch.Size([3, 1, 1, 3]) modify the mask
# scores torch.Size([5, 8, 17, 3])  --> upsample it



# ===================================== Video based model ==================================== #
# Implementation is from https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

# from mmcv.runner import load_checkpoint
# from mmaction.utils import get_root_logger
# from ..builder import BACKBONES

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x



# x shape torch.Size([1, 1, 56, 56, 128]) = B, D, H, W, C,  x_size = (D, H, W) = (1, 56, 56)
def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i] # 2 -> 1
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# class WindowAttention3D(nn.Module):
#     """ Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The temporal length, height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wd, Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_d = torch.arange(self.window_size[0])
#         coords_h = torch.arange(self.window_size[1])
#         coords_w = torch.arange(self.window_size[2])
#         coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 2] += self.window_size[2] - 1

#         relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
#         relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
#         relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """ Forward function.
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, N, N) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
#             N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# Implementation is from https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
# class SwinTransformerBlock3D(nn.Module):
#     """ Swin Transformer Block.
#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         window_size (tuple[int]): Window size.
#         shift_size (tuple[int]): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.use_checkpoint=use_checkpoint

#         assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
#         assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
#         assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention3D(
#             dim, window_size=self.window_size, num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward_part1(self, x, mask_matrix):
#         B, D, H, W, C = x.shape
#         window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

#         x = self.norm1(x)
#         # pad feature maps to multiples of window size
#         pad_l = pad_t = pad_d0 = 0
#         pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
#         pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
#         pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
#         x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
#         _, Dp, Hp, Wp, _ = x.shape
#         # cyclic shift
#         if any(i > 0 for i in shift_size):
#             shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
#             attn_mask = mask_matrix
#         else:
#             shifted_x = x
#             attn_mask = None
#         # partition windows
#         x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
#         # merge windows
#         attn_windows = attn_windows.view(-1, *(window_size+(C,)))
#         shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
#         # reverse cyclic shift
#         if any(i > 0 for i in shift_size):
#             x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
#         else:
#             x = shifted_x

#         if pad_d1 >0 or pad_r > 0 or pad_b > 0:
#             x = x[:, :D, :H, :W, :].contiguous()
#         return x

#     def forward_part2(self, x):
#         return self.drop_path(self.mlp(self.norm2(x)))

#     def forward(self, x, mask_matrix):
#         """ Forward function.
#         Args:
#             x: Input feature, tensor size (B, D, H, W, C).
#             mask_matrix: Attention mask for cyclic shift.
#         """

#         shortcut = x
#         if self.use_checkpoint:
#             x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
#         else:
#             x = self.forward_part1(x, mask_matrix)
#         x = shortcut + self.drop_path(x)

#         if self.use_checkpoint:
#             x = x + checkpoint.checkpoint(self.forward_part2, x)
#         else:
#             x = x + self.forward_part2(x)

#         return x


class SwinMLPBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        # self.attn = WindowAttention3D(
        #     dim, window_size=self.window_size, num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # ===================================================================== #
        # use group convolution to implement 3D multi-head MLP
        # NEED modification
        # in temporal case,         input is  B*nW, nH*Wd*Wh*Ww, C//nH
        # in non temporal case, the input is  nW*B, nH*window_size*window_size, C//nH

        # # non-temporal case
        # self.spatial_mlp_3D = nn.Conv1d(self.num_heads * self.window_size ** 2,
        #                              self.num_heads * self.window_size ** 2,
        #                              kernel_size=1,
        #                              groups=self.num_heads)

        # # x shape torch.Size([1, 1, 56, 56, 128]) = B, D, H, W, C,  x_size = (D, H, W) = (1, 56, 56)
        # def get_window_size(x_size, window_size, shift_size=None):
        #     use_window_size = list(window_size)
        #     if shift_size is not None:
        #         use_shift_size = list(shift_size)
        #     for i in range(len(x_size)):
        #         if x_size[i] <= window_size[i]:
        #             use_window_size[i] = x_size[i] # 2 -> 1
        #             if shift_size is not None:
        #                 use_shift_size[i] = 0

        #     if shift_size is None:
        #         return tuple(use_window_size)
        #     else:
        #         return tuple(use_window_size), tuple(use_shift_size)


        # # temporal case
        self.spatial_mlp_3D = nn.Conv1d(self.num_heads * self.window_size[0] * self.window_size[1] * self.window_size[2],
                                     self.num_heads * self.window_size[0] * self.window_size[1] * self.window_size[2],
                                     kernel_size=1,
                                     groups=self.num_heads)        
        # ===================================================================== #

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape # batchsize, frame_length, input resolution h, input resolution w, dimension
        # print('x shape', x.shape) # x shape torch.Size([1, 1, 56, 56, 128]) x shape torch.Size([3, 2, 14, 14, 512])
        
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # print('window_size after get_window_size', window_size) # window_size after get_window_size (1, 7, 7)
        # window_size after get_window_size (2, 14, 14)
        # window_size after get_window_size (2, 14, 14)
        # window_size after get_window_size (2, 7, 7)
        #RuntimeError: Given groups=32, weight of size [12544, 392, 1], expected input[1, 3136, 32] to have 12544 channels, but got 3136 channels instead

        
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        
        # #  W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        
        # # merge windows
        # attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        # shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # ================== Window/Shifted-Window 3D Spatial MLP ====================== #
        x_windows = x_windows.view(-1, window_size[0] * window_size[1]* window_size[2], C)
        x_windows_heads = x_windows.view(-1, window_size[0] * window_size[1]* window_size[2], self.num_heads, C//self.num_heads)
        # print('x1', x_windows_heads.shape) # x1 torch.Size([1, 49, 32, 32])
        x_windows_heads = x_windows_heads.transpose(1, 2)  # B*nW, nH, Wd*Wh*Ww, C//nH
        # print('x2', x_windows_heads.shape) # x2 torch.Size([1, 32, 49, 32])
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * window_size[0] * window_size[1]* window_size[2],
                                                  C // self.num_heads) # B*nW, nH*Wd*Wh*Ww, C//nH
        spatial_mlp_windows = self.spatial_mlp_3D(x_windows_heads) # B*nW, nH*Wd*Wh*Ww, C//nH
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, window_size[0] * window_size[1]* window_size[2],
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, window_size[0] * window_size[1]* window_size[2], C) # # B*nW, Wd*Wh*Ww, C

        # merge windows
        spatial_mlp_windows = spatial_mlp_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(spatial_mlp_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        
        # ============================================================================================= #
        
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
# @lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

# Here, the 3D patch is A cube 
class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size() # x0, input of the encoder torch.Size([1, 4, 3, 224, 224])
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:  # self.patch_size[0] is the about time axis, is sequence_length
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class SwinMLP3D_Encoder(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 d_model,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        print('================ Check the version of the model ==========')
        print('embed_dim', embed_dim)
        print('depths', depths)
        print('num_heads', num_heads)
        print('patch_size', patch_size)
        print('window_size', window_size)
        print('===========================================================')

        self.d_model = d_model
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self.head = nn.Linear(self.num_features, self.d_model)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # print('x0, input of the encoder', x.shape) # x0, input of the encoder torch.Size([1, 3, 3, 224, 224]) ; x0, input of the encoder torch.Size([1, 4, 3, 224, 224])
        # we hope it is torch.Size([1, 3, 4, 224, 224])
        # # x0, input of the encoder torch.Size([3, 3, 4, 224, 224])
        # new torch.Size([3, 4, 3, 224, 224])

        # =========== mengya: prepare proper input shape for Swin 3D modeL ==================== #
        #(36, 3, 224, 224) --> (9, 4, 3, 224, 224) --> (9, 3, 4, 224, 224) --> 3gpus, each gpu has torch.Size([3, 3, 4, 224, 224])
        # x = rearrange(x, 'n c d h w -> n d h w c')
        x = rearrange(x, 'n d i h w -> n i d h w') # x0, input of the encoder torch.Size([3, 3, 4, 224, 224])
        # ====================================================================================== #
    
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        # print('x0 after 3D patch embedding', x.shape) # x0 after 3D patch embedding torch.Size([1, 96, 1, 56, 56])

        for layer in self.layers:
            x = layer(x.contiguous())
        
        # print('x1 inside swin3D model', x.shape) # x1 inside swin3D model torch.Size([1, 768, 1, 7, 7])

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        # print('x2 inside swin3D model', x.shape) # x2 inside swin3D model torch.Size([1, 1, 7, 7, 768]) # expect it will be converted into [1, 1*1*7, 768]
        # x2 inside swin3D model torch.Size([3, 1, 7, 7, 768]

        # x = rearrange(x, 'n d h w c -> n c d h w')
        # print('x3 inside swin3D model', x.shape) # x3 inside swin3D model torch.Size([1, 768, 1, 7, 7])
        
        # ====== mengya: added for captioning case ============= #
        x = rearrange(x, 'n d h w c -> n (d h w) c')
        # print('x4 output of encoder', x.shape) # x4 output of encoder torch.Size([1, 49, 768])
        x = self.head(x)
        # print('x5 output of encoder', x.shape) # x5 output of encoder torch.Size([1, 49, 512])
        # x5 output of encoder torch.Size([3, 49, 512])
        # x5 output of encoder torch.Size([3, 98, 512]) when using patch_size=(2,4,4)  98=7*7*2  4/2=2
        # =======================================================#


        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinMLP3D_Encoder, self).train(mode)
        self._freeze_stages

# ============================================================================================================================== #

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Video_SwinMLP_TranCAP_L(AttModel):

    def make_model(self, opt, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        # Added by mengya
        self.opt = opt
        c = copy.deepcopy
        # original one
        attn = MultiHeadedAttention(h, d_model, dropout)

        # # replace attn with Pooling
        # attn = Pooling(pool_size=3)

        # # # replace attn with Swin Transformer Window based multi-head self attention
        # attn = WindowAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # # original Transformer model: def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        # model = EncoderDecoder(
        #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
        #     Decoder(DecoderLayer(d_model, c(attn), c(attn), 
        #                          c(ff), dropout), N_dec),
        #     lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #     nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        #     Generator(d_model, tgt_vocab))


        # https://github.com/microsoft/Swin-Transformer/tree/3dc2a55301a02d71e956b382b50943a35c4abee9/configs
        # base configuration:EMBED_DIM: 128, DEPTHS: [ 2, 2, 18, 2 ], NUM_HEADS: [ 4, 8, 16, 32 ], WINDOW_SIZE: 7
        # test window_size=(1,7,7) firstly
        model = EncoderDecoder(
            SwinMLP3D_Encoder(
                   d_model=self.opt.d_model,
                   pretrained=None,
                   pretrained2d=True,
                   patch_size=(2,4,4),
                   in_chans=self.opt.in_channels,
                   embed_dim=self.opt.dim,
                   depths=[2, 2, 18, 2 ],
                   num_heads=[4, 8, 16, 32 ],
                   window_size=(2,7,7),
                   mlp_ratio=4.,
                   qkv_bias=True,
                   qk_scale=None,
                   drop_rate=0.,
                   attn_drop_rate=0.,
                   drop_path_rate=0.1,
                   norm_layer=nn.LayerNorm,
                   patch_norm=self.opt.patch_norm,
                   frozen_stages=-1,
                   use_checkpoint=False), # Default: patch_norm= False, patch_size=(4,4,4), window_size=(2,7,7)

            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N_dec),
            lambda x:x, 
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(Video_SwinMLP_TranCAP_L, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        # self.att_embed = nn.Sequential(*(
        #                             ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
        #                             (nn.Linear(self.att_feat_size, self.d_model),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))+
        #                             ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))


        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.att_feat_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1


        # self.model = self.make_model(0, tgt_vocab,
        #     N_enc=self.N_enc,
        #     N_dec=self.N_dec,
        #     d_model=self.d_model,
        #     d_ff=self.d_ff,
        #     h=self.h,
        #     dropout=self.dropout)

        self.model = self.make_model(self.opt,
            0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)


        # # # =================== patch related code ==================== #
        # # split image into non-overlapping patches
        # # from swin transformer's patch embedding
        # img_size = self.opt.image_size
        # patch_size = self.opt.patch_size
        # in_chans = self.opt.in_channels
        # embed_dim = self.opt.dim
        # drop_rate = self.opt.emb_dropout
        # norm_layer = nn.LayerNorm
        # self.patch_norm = self.opt.patch_norm
        # self.ape = self.opt.ape

        # print('self.patch_norm', self.patch_norm, 'self.ape: ', self.ape)

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        
        # num_patches = self.patch_embed.num_patches
        # # print('num_patches', num_patches) # 3136

        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)


        # # added by mengya
        # self.att_mask_embed = nn.Linear(num_patches, self.d_model)
        # # # ================================================================== #

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    # def _prepare_feature(self, fc_feats, att_feats, att_masks):
    #     att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
    #     # memory = self.model.encode(att_feats, att_masks)
    #     # for swin transformer encoder
    #     memory = self.model.encode(att_feats)

    #     return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    # # _prepare_feature will be called for beam search in AttModel.py
    # def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
    #     # ============== patch related code ===================== #
    #     att_feats = self._prepare_patches_forward(att_feats) # x after patch embedding torch.Size([5, 3136, 96])
    #     # print('x2 after patch embedding', att_feats.shape) # x2 after patch embedding torch.Size([5, 3136, 96])
    #     # print('att_feats type', att_feats.type()) # att_feats type torch.cuda.FloatTensor
    #     # ================================================================== #
    #     # why M2T Transformer prepare the att_mask so easily: https://github.com/aimagelab/meshed-memory-transformer/blob/master/models/transformer/encoders.py
    #     # attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
    #     # 处理非定长序列

    #     att_feats, att_masks = self.clip_att(att_feats, att_masks)

    #     att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

    #     # print('att_feats after self.att_embed', att_feats.shape) # att_feats after self.att_embed torch.Size([5, 3136, 96])

    #     if att_masks is None:
    #         att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long) # # att_masks torch.Size([5, 3136])
    #     att_masks = att_masks.unsqueeze(-2) # # att_masks torch.Size([5, 1, 3136])

    #     # ================================================== #
    #     # print('att_masks type', att_masks.type()) # att_masks type torch.cuda.LongTensor
    #     # Added by mengya for decoder
    #     att_masks = att_masks.type(torch.cuda.FloatTensor)
    #     # print('att_masks type 2', att_masks.type())
    #     att_masks = self.att_mask_embed(att_masks) # RuntimeError: "addmm_cuda" not implemented for 'Long', src_mask in decoder torch.Size([5, 1, 256])
    #     # ================================================== #


    #     if seq is not None:
    #         # crop the last one
    #         # seq = seq[:,:-1]
    #         seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
    #         seq_mask[:,0] = 1 # bos

    #         seq_mask = seq_mask.unsqueeze(-2)
    #         seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

    #         seq_per_img = seq.shape[0] // att_feats.shape[0]
    #         if seq_per_img > 1:
    #             att_feats, att_masks = utils.repeat_tensors(seq_per_img,
    #                 [att_feats, att_masks]
    #             )
    #     else:
    #         seq_mask = None

    #     return att_feats, seq, att_masks, seq_mask
    
    # def _prepare_patches_forward(self, x):
    #     # print('x in _prepare_patches_forward', x.shape) # x in _prepare_patches_forward torch.Size([5, 3, 224, 224])
    #     x = self.patch_embed(x)
    #     if self.ape:
    #         x = x + self.absolute_pos_embed
    #     x = self.pos_drop(x)
    #     # print('x after patch embedding', x.shape) # x after patch embedding torch.Size([5, 3136, 96])
    #     return x

    # def _forward(self, fc_feats, att_feats, seq, att_masks=None):
    #     if seq.ndim == 3:  # B * seq_per_img * seq_len
    #         seq = seq.reshape(-1, seq.shape[2])
    #     att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        
    #     # out = self.model(att_feats, seq, att_masks, seq_mask) # some error cannot solve. so just disable att_masks.
    #     out = self.model(att_feats, seq, None, seq_mask)

    #     outputs = self.model.generator(out)
    #     return outputs

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        seq, seq_mask = self._prepare_feature_forward(att_feats) # Here, att_feats is (batch_size, 3, 224, 224). seq will be None as the default setting
        memory = self.model.encode(att_feats)
        
        # print('att_feats for beam search', att_feats.shape) # att_feats for beam search torch.Size([5, 3, 224, 224])
        # print('att_feats[...,:0]', att_feats[...,:0].shape) # torch.Size([5, 3, 224, 0])
        
        # att_feats[...,:0] convert (50, 196, 2048) (50, 196, 0)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, seq=None):
        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return seq, seq_mask


    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        # att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(seq)
        seq, seq_mask = self._prepare_feature_forward(att_feats, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = self.model.generator(out)
        return outputs


    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))

        # # when decoder did not use att_mask
        # out = self.model.decode(memory, None, 
        #                        ys, 
        #                        subsequent_mask(ys.size(1))
        #                                 .to(memory.device))

        return out[:, -1], [ys.unsqueeze(0)]

# ****************************************************************************************** #

# scores = scores.masked_fill(mask == 0, float('-inf'))

# # about the demo code for the mask:
# >>> import torch
# >>> a=torch.tensor([[[5,5,5,5], [6,6,6,6], [7,7,7,7]], [[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
# >>> mask = torch.ByteTensor([[[1],[1],[0]],[[0],[1],[1]]])
# >>> print(mask==0)
# tensor([[[False],
#          [False],
#          [ True]],

#         [[ True],
#          [False],
#          [False]]])
# >>> b = a.masked_fill(mask, value=torch.tensor(-1e9)

# >>> c = a.masked_fill(mask==0, value=torch.tensor(-1e9))

# >>> print('a', a)
# a tensor([[[5, 5, 5, 5],
#          [6, 6, 6, 6],
#          [7, 7, 7, 7]],

#         [[1, 1, 1, 1],
#          [2, 2, 2, 2],
#          [3, 3, 3, 3]]])
# >>> print('b', b)
# b tensor([[[-1000000000, -1000000000, -1000000000, -1000000000],
#          [-1000000000, -1000000000, -1000000000, -1000000000],
#          [          7,           7,           7,           7]],

#         [[          1,           1,           1,           1],
#          [-1000000000, -1000000000, -1000000000, -1000000000],
#          [-1000000000, -1000000000, -1000000000, -1000000000]]])

# >>> print('c', c)
# c tensor([[[          5,           5,           5,           5],
#          [          6,           6,           6,           6],
#          [-1000000000, -1000000000, -1000000000, -1000000000]],

#         [[-1000000000, -1000000000, -1000000000, -1000000000],
#          [          2,           2,           2,           2],
        #  [          3,           3,           3,           3]]])



'''
# ===================================== Swin Transformer (Start) ===================================================== #
        model = EncoderDecoder(
            SwinMLP3D_Encoder(
                   d_model=self.opt.d_model,
                   pretrained=None,
                   pretrained2d=True,
                   patch_size=(4,4,4),
                   in_chans=self.opt.in_channels,
                   embed_dim=self.opt.dim,
                   depths=[2, 2, 18, 2 ],
                   num_heads=[4, 8, 16, 32 ],
                   window_size=(1,7,7),
                   mlp_ratio=4.,
                   qkv_bias=True,
                   qk_scale=None,
                   drop_rate=0.,
                   attn_drop_rate=0.,
                   drop_path_rate=0.1,
                   norm_layer=nn.LayerNorm,
                   patch_norm=self.opt.patch_norm,
                   frozen_stages=-1,
                   use_checkpoint=False), # Default: patch_norm= False, patch_size=(4,4,4), window_size=(2,7,7)

# swin_transformer WindowAttention: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Swin transformer video tutorial: https://www.youtube.com/watch?v=tFYxJZBAbE8

# A single layer of transformer is replaced by two layers W-MSA and SW-MSA. W stands for window, SW stands for shifted window. 
# Shifted window attention: W-MSA (Window-based Multi-Head Self Attention) is where we divide the input image into four windows and compute attention for patches within the window. This happens in the
# first layer of the swin transformer block. The next stage if shifted window MSA.
# In convolutional neural networks, there's a kernel that slides along the image and computes the outputs but something like that is missing in transformers.
# So in the second layer, they propose to shift the window by two patches and compute the attention within these windows. 
# For empty space without any pixels, a naive solution is to zero pad that space. A mor nice solution is "cycle shifting": copy over the patches on top to the bottom, and from left to right, and also
# diagonally across to make up for the missing patches.

# when patch size=4, the dimension of each batch is 4*4*3=48. Thus, the input to the swin transformer is (224/4 * 224/4 * 48)
# 48 --> 96, 128, 192
# What's Patch Merging? A simple image composed of 8*8 pixels, each patch in the image is a 4*4 pixels. In terms of patches, I have two by two patches now. In patch merging, I take the two by two
# neighboring patches, merge or combine them together. So that I get one patch in place of the two by two patches. (2*2 patches --> 1*1 patch)
# How to implement patch merging? Use a linear layer. we have the ability to define the size of the output we want. 
# The main intuition is that we have decrease the number of patches by 2, but we have increase C by 2 as well. 

# The drawback of MSA in classic transformer model: Attention calculation is very compute heavy.
# Specifically, we have to compute the self-attention for a given patch with all the other patches in the input image.
# so clearly this becomes very compute intensive even for a reasonably large sized image. 
# To overcome it, swin tranformer introduces "window". A window divides the input image into serveral parts. window size vs. (larger than) patch size.
# After dividing into such window, whenever we compute self attention between patches within that window, and we ignore the rest of the patches.


# when patch_size = 8. the dimension of each batch is 8*8*3=192. Thus, the input to the swin transformer is (28, 28, 192)
# 192 --> 

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# The input image is divided into serveral windows.
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # mengya： It supports both of shifted and non-shifted window. e.g. W-MSA and SW-MSA
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # mengya: the first LN is applied here.
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # mengya: compute self attention between patches within that window, and we ignore the rest of the patches.
        # mengya: the att_mask/mask here is self.attn_mask
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C  

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # MLP is applied here for classification case? but we need to change it for our captioning case

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    # mengya: BasicLayer includes 2 SwinTransformerBlock + Patch Merging
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x) # mengya: downsample is actually Patch Merging layer.
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths) # 一个虚线框代表一个layer, self.num_layers = 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 除了第一个虚线框，之后每经过一个虚线框，都乘2
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint) # 4个虚线框(4个BasicLayer)中，有3个BasicLayer需要 “downsample=PatchMerging“
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
# ============================================ Swin Transformer (End) =========================================================== #
'''