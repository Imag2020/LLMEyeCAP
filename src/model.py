import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import os
import json
import numpy as np
from collections import defaultdict
import random
from tqdm.notebook import tqdm
from torchvision import models
from torch.nn.utils.rnn import pad_sequence
import matplotlib.patches as patches

import math
import time
import os
from PIL import Image
import requests
import nltk

import os
import cv2
import colorsys
from numpy import asarray
import math


from transformers import GPT2LMHeadModel, GPT2Config

from scipy.optimize import linear_sum_assignment

import sys
sys.path.append("../src")

from utils import *

NUM_QUERIES = 40
feature_size = 256  # Pour ResNet50
token_size = 256  # Pour GPT-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# minimal updates here 

"""
Various positional encodings for the transformer.
"""

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    '''
    The line mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] applies a mask to the output
    features from the backbone. The mask is used to indicate which pixels in the image are valid.


    The mask is a tensor of the same size as the output features. The mask is initialized to all zeros. The m[None].float() 
    operation expands the mask to be a 1-D tensor of size 1 x H x W. The F.interpolate() 
    operation then resizes the mask to the same size as the output features. The to(torch.bool) operation converts the
    mask to a binary tensor. The [0] operation takes the first element of the tensor, which is the mask for the first output 
    feature map.

    The mask of a feature extracted from ResNet50 as a backbone is a binary tensor that indicates which pixels in the image 
    are valid. The pixels that are valid are those that are not padded. The mask is used by the backbone to ignore the padded 
    pixels when it is extracting features from the image.

    '''

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
            # ==> todo weights=ResNet50_Weights.DEFAULT)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def get_sinusoid_encoding_table(n_position, d_hid):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

class Parameters:
    def __init__(self):
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 2
        self.weight_decay = 1e-4
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1

args = Parameters()

args.lr=1e-4
args.lr_backbone=1e-5
args.batch_size=32
args.weight_decay=1e-4
args.epochs=300
args.lr_drop=200
args.clip_max_norm=0.1 # type=float,    help='gradient clipping max norm')

# Model parameters
args.frozen_weights=False # ', type=str, default=None, #    help="Path to the pretrained model. If set, only the mask head will be trained")

# * Backbone
args.backbone='resnet50' # type=str, #     help="Name of the convolutional backbone to use")
args.dilation=False  # ', action='store_true',          #      help="If true, we replace stride with dilation in the last convolutional block (DC5)")
args.position_embedding='sine' # type=str, choices=('sine', 'learned'),     help="Type of positional embedding to use on top of the image features")

# * Transformer
args.enc_layers=6 # type=int,      help="Number of encoding layers in the transformer")
args.dec_layers=6 # type=int,       help="Number of decoding layers in the transformer")
args.dim_feedforward=2048  # ===> type=int,   help="Intermediate size of the feedforward layers in the transformer blocks")
args.hidden_dim=256  # ===> type=int,   help="Size of the embeddings (dimension of the transformer)")
args.dropout=0.1   #type=float,   help="Dropout applied in the transformer")
args.nheads=8   #type=int,   help="Number of attention heads inside the transformer's attentions")
args.num_queries=40  #type=int,   help="Number of query slots")
args.pre_norm=True # ', action='store_true')

# * Segmentation
args.masks=False #, action='store_true',  help="Train segmentation head if the flag is provided")


"""
LLMEyeCap Transformer class.

A DETR (FaceBook) Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

"""
import copy
from typing import Optional, List


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class LLMEyeCap(nn.Module): # Im Novel Object Captioning V 0.1 
    
    def __init__(self, backbone, transformer, num_queries, vocab_size,pad_token):
        
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        
        self.caption_embed = nn.Linear(self.hidden_dim, vocab_size)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        '''
        self.capdecoder =  CaptioningDecoder(detr_decoder_dim=transformer.d_model, token_embedding_dim=transformer.d_model, 
                                             vocab_size=vocab_size, num_queries=num_queries, num_layers=6)
        '''
        self.capdecoder = CaptionDecoder(feature_size, token_size, vocab_size,num_queries,pad_token ).to(device)
        

    def forward(self, samples: NestedTensor, captions):
                            
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)  #featers + position embedding 
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]    
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        outputs_captions=self.capdecoder(hs,captions)
        # predicted_sequences = torch.argmax(outputs_captions, dim=-1)

        out = {'pred_logits': outputs_captions , 'pred_boxes': outputs_coord[-1]}        
        return out
    
    def generate_caption(self, image_path, tokenizer, max_length, pad_sos):
        
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        image = transform(image).unsqueeze(0).to(device)
        
        if isinstance(image, (list, torch.Tensor)):
            image = nested_tensor_from_tensor_list(image)
        
        with torch.no_grad():
            features, pos = self.backbone(image)  #featers + position embedding 
            src, mask = features[-1].decompose()
            assert mask is not None
            
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]    
            outputs_coord = self.bbox_embed(hs).sigmoid()
            
            input_ids = torch.ones((1, 40, 1), dtype=torch.long, device=device)
            input_ids.fill_(pad_sos)

            
            for i in range(max_length):
                outputs_captions = self.capdecoder(hs, input_ids)
                predicted_sequences = torch.argmax(outputs_captions, dim=-1)
                next_token = predicted_sequences[:, :, -1:]  # take the last token from the sequence
                input_ids = torch.cat((input_ids, next_token), dim=-1)

            #caption = tokenizer.detokenize(input_ids[0].tolist()) #, skip_special_tokens=True)

        return outputs_coord[-1], input_ids # caption[-1]

class LLMEyeCapModel(nn.Module):
    def __init__(self, num_queries,vocab_size,pad_token):
        super(LLMEyeCapModel,self).__init__()
        self.num_queries = num_queries
        self.vocab_size=vocab_size
        self.backbone = build_backbone(args)
        self.transformer = build_transformer(args)

        self.model = LLMEyeCap(
        self.backbone,
        self.transformer,
        num_queries=self.num_queries,
        vocab_size=self.vocab_size,
        pad_token=pad_token
        ) 
        
        # self.in_features = self.caption_embed.in_features   
        
        # self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        
        self.model.num_queries = self.num_queries
        
    def forward(self,images,captions):
        return self.model(images,captions)
    
    def generate_caption(self, image_path, tokenizer, max_length=20,pad_sos=0):
        return self.model.generate_caption(image_path, tokenizer, max_length,pad_sos)

class CaptionDecoder(nn.Module):
    def __init__(self, detr_decoder_dim, token_embedding_dim, vocab_size, num_queries, pad_token, num_layers=6):
        super(CaptionDecoder, self).__init__()
        
        self.detr_decoder_dim = detr_decoder_dim
        self.token_embedding_dim = token_embedding_dim
        self.vocab_size = vocab_size
        self.num_queries = num_queries
        self.pad_token = pad_token

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, token_embedding_dim)
        
        # Initialize GPT-2
        config = GPT2Config(vocab_size=vocab_size, n_embd=detr_decoder_dim + token_embedding_dim, n_head=8 )
        self.gpt2 = GPT2LMHeadModel(config)
        
        self.target_projection = nn.Linear(token_embedding_dim, detr_decoder_dim + token_embedding_dim)
        
    def forward(self, detr_output, captions):
        
        
        # Create an attention mask with shape [batch_size, num_queries, sequence_length]
        attention_mask = (captions != self.pad_token).float().to(captions.device)  # [batch_size, num_queries, sequence_length]


        seq_length = captions.size(2)
        pos_encoding = get_sinusoid_encoding_table(seq_length, self.token_embedding_dim).to(captions.device)
        pos_encoding = pos_encoding.unsqueeze(0).repeat(captions.size(0) * self.num_queries, 1, 1)
        
        # Get the last layer's output from the DETR decoder
        spatial_embedding = detr_output[-1]  # [batch_size, num_queries, detr_decoder_dim]
        
        # Get token embeddings
        token_embeddings = self.token_embedding(captions)  # [batch_size, num_queries, seq_length, token_embedding_dim]
        
        # Repeat the spatial embedding for each token in the sequence and concatenate
        spatial_embedding = spatial_embedding.unsqueeze(2)  # Add seq_length dimension: [batch_size, num_queries, 1, detr_decoder_dim]
        combined_embedding = torch.cat([spatial_embedding.repeat(1, 1, token_embeddings.size(2), 1), token_embeddings], dim=-1)
        # combined_embedding shape: [batch_size, num_queries, seq_length, detr_decoder_dim + token_embedding_dim]
        
        # Prepare the memory for the transformer decoder
        memory = combined_embedding.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.detr_decoder_dim + self.token_embedding_dim)
        # memory shape: [seq_length, batch_size*num_queries, detr_decoder_dim + token_embedding_dim]
        
        # Prepare the target for the transformer decoder (using token embeddings)
        target = token_embeddings.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.token_embedding_dim)
        # target shape: [seq_length, batch_size*num_queries, token_embedding_dim]
        
        
        pos_encoding = pos_encoding.permute(1, 0, 2)
        target += pos_encoding


        # Project target to the required dimension
        
        target = self.target_projection(target)
        
        attention_mask = attention_mask.permute(2, 0, 1).reshape(captions.size(2), -1)
        tgt_key_padding_mask = (attention_mask == 0).permute(1,0)
        
        # Prepare the inputs for GPT-2
        inputs_embeds = combined_embedding.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.detr_decoder_dim + self.token_embedding_dim)
    
        # Reshape attention_mask for GPT-2. Flatten the batch_size and num_queries dimensions.
        attention_mask = attention_mask.reshape(-1, captions.size(2))  # New shape: [batch_size * num_queries, sequence_length]
    
        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
    
        # Reshape logits to match the original shape
        logits = logits.view(captions.size(2), captions.size(0), self.num_queries, self.vocab_size).permute(1, 2, 0, 3)
    
        return logits

