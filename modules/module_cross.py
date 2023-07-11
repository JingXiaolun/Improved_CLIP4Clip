from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

from visualizer import get_local

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict

from .helpers import complement_idx

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    @get_local('attn')
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # attn shape: [32, 8, 13, 13] | attn_mask shape: [32, 13, 13] | attn_mask_ shape: [32, 8, 13, 13]
        if attn_mask is not None:
            attn_mask_ = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn.masked_fill(attn_mask_ == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        ################################################# save attn in .pt format ##############################################################################################
        ## attn_complete.pt refers to attention between compelete sequence, attn_pruning.pt refers to attention between pruning sequence
        #prefix_dir = '/public/home/jinxl/jinxl/research/Experiment/Improved_CLIP4CLip/Code/visualize/attention'
        #if attn.shape[-2]==13:
        #    attn_path = prefix_dir + '/' + 'attn_complete.pt'
        #else:
        #    attn_path = prefix_dir + '/' + 'attn_pruning.pt'
        #torch.save(attn, attn_path)
        ################################################# save attn in .pt format ##############################################################################################

        keep_rate = self.keep_rate
        left_tokens = N - 1
        if keep_rate < 1:  
            left_tokens_boundary = math.ceil(keep_rate * (N - 1))
            assert left_tokens_boundary >= 1
            if left_tokens_boundary == N - 1:
                return x, None, None, left_tokens_boundary

            idx_list, index_list = [], []    
            for i in range(B):
                # shape: [H, N, N]
                sub_attn_mask_= attn_mask_[i]
                # shape: [N-1]
                sub_attn_mask_row = sub_attn_mask_[0, 0, 1:]

                # shape: [H, N-1]
                sub_cls_attn_all = attn[i][:, 0, 1:]

                # shape: [H, M]
                valid_index = (sub_attn_mask_row!=0).expand(self.num_heads, -1)

                # shape: [H, M] -> [M]
                sub_cls_attn_valid = sub_cls_attn_all[valid_index].view(self.num_heads, -1)
                sub_cls_attn_valid = sub_cls_attn_valid.mean(dim=0) 
                valid_tokens_num = sub_cls_attn_valid.shape[0]

                left_tokens = math.ceil(valid_tokens_num * keep_rate)

                # sub_idx: torch.tensor([2, 6, 9, 4, 1, 7]) (shape: [left_tokens])
                _, sub_idx = torch.topk(sub_cls_attn_valid, left_tokens, dim=0, largest=True, sorted=True)  # [left_tokens]

                left_tokens_remaining = left_tokens_boundary - left_tokens
                sub_idx_remaining = (torch.ones(left_tokens_remaining)*1e9).to(sub_idx.device)
                sub_idx_padding = torch.cat([sub_idx, sub_idx_remaining], dim=0) # torch.tensor([2, 6, 9, 4, 1, 7, 1e9, 1e9]) (e.g. M=10, keep_rate=0.6)
                assert sub_idx_padding.shape[0] == left_tokens_boundary, 'the dimension of idx must be same through padding'
                    
                sub_idx = sub_idx_padding.unsqueeze(0) # shape: [1, left_tokens_boundary]
                sub_index = sub_idx_padding.unsqueeze(0).unsqueeze(-1).expand(-1, -1, C) # shape: [1, left_tokens_boundary, C]

                idx_list.append(sub_idx)
                index_list.append(sub_index)

            idx = torch.cat(idx_list, dim=0) # shape: [B, left_tokens_boundary]
            index = torch.cat(index_list, dim=0) # shape: [B, left_tokens_boundary, C]
            return x, index, idx, left_tokens_boundary
        return  x, None, None, left_tokens

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, keep_rate: float, fuse_token: bool):
        super().__init__()
        self.attn = Attention(dim=d_model, num_heads=n_head, keep_rate=keep_rate)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head
        self.keep_rate = keep_rate
        self.fuse_token = fuse_token
    
        self.padding_operation = nn.ConstantPad2d((1, 0, 1, 0), 1)

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        x, attn_mask = para_tuple
      
        # step1: get output from self.attn and implement residual operation 
        '''
        tmp: shape of [B, N, C]
        index: shape of [B, left_tokens_boundary, C]
        idx: shape of [B, left_tokens_boundary]
        left_tokens: shape of [1] 
        '''
        tmp, index, idx, left_tokens_boundary = self.attn(self.ln_1(x), attn_mask)
        x = x + tmp

        # step2: tokens removal 
        if index is not None:
            B, N, C = x.shape

            sub_x_list = []
            attn_mask_flag_list = []
            for i in range(B):
                sub_x = x[i]  # shape: [N, C]
                sub_idx = idx[i]  # shape: [left_tokens_boundary]
                sub_index = index[i]  # shape: [left_tokens_boundary, C]
                sub_gather_index = sub_index[sub_index<1e4].view(-1, C) # shape: [valid_token_num, C]

                invalid_token_num = torch.sum(sub_idx > 1e4).item()
                if not invalid_token_num:
                    # shape: [N-1, C]
                    sub_x_non_cls = sub_x[1:]
                else:
                    # shape: [N-1-invalid_token_num, C]
                    sub_x_non_cls = sub_x[1:-invalid_token_num]

                sub_x_others = torch.gather(sub_x_non_cls, dim=0, index=sub_gather_index.type(torch.int64))
                sub_x_remaining = torch.zeros(left_tokens_boundary - sub_x_others.shape[0]).unsqueeze(-1).expand(-1, C)
                sub_x_remaining = sub_x_remaining.to(sub_x_others.device)

                sub_x_noncls = torch.cat([sub_x_others, sub_x_remaining], dim=0)  # shape: [left_tokens_boundary, C]
                attn_mask_flag_list.append((sub_x_noncls==0)[:, 0].unsqueeze(0))  # shape: [1, left_tokens_boundary]

                sub_x_ = torch.cat([sub_x[0].unsqueeze(0), sub_x_noncls], dim=0)  # shape: [1 + left_tokens_boundary, C]
                sub_x_ = sub_x_.unsqueeze(0)  # shape: [1, 1 + left_tokens_boundary, C]

                sub_x_list.append(sub_x_)

            x = torch.cat(sub_x_list, dim=0)  # shape: [B, 1 + left_tokens_boundary, C]
            attn_mask_flag = torch.cat(attn_mask_flag_list, dim=0)  # shape: [B, left_tokens_boundary]

            ####################################################################################################################
            #if self.fuse_token:
            #    compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
            #    non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

            #    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
            #    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
            #    x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            #else:
            #    x = torch.cat([x[:, 0:1], x_others], dim=1)
            ####################################################################################################################

        # step3: LayerNorm + MLP (Residual Operation)
        x = x + self.mlp(self.ln_2(x))

        # step4: update attn_mask (corresponding to purned visual output)
        if index is not None:
            attn_mask_new = torch.ones(B, left_tokens_boundary) # shape: [B, left_tokens_boundary]
            attn_mask_new = attn_mask_new.to(x.device)

            attn_mask_new[attn_mask_flag] = 0  # shape: [B, left_tokens_boundary]
            attn_mask_new = attn_mask_new.unsqueeze(1).expand(-1, left_tokens_boundary, -1)  # [B, left_tokens_boundary, left_tokens_boundary]
        
            ## padding constant in top and left corner to introduce cls_token
            attn_mask_new = self.padding_operation(attn_mask_new)  # [B, left_tokens_boundary + 1, left_tokens_boundary + 1]
    
            for i in range(B):
                sub_attn_mask_new = attn_mask_new[i]  # shape: [left_tokens_boundary + 1, left_tokens_boundary + 1]
                if True in (sub_attn_mask_new==0):
                    attn_mask_new[i, 0, 1:][sub_attn_mask_new[1, 1:]==0] = 0
            attn_mask = attn_mask_new  # shape: [B, left_tokens_boundary + 1, left_tokens_boundary + 1]
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, keep_rate: float, fuse_token: bool):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, keep_rate, fuse_token) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        for i in range(self.layers):
            (x, attn_mask) = self.resblocks[i]((x, attn_mask))
        return (x, attn_mask)

class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CrossModel(PreTrainedModel):

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__(config)

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads,)
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output
