#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        #add for backdoor
        self.on=True

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        self.prompt_dim=prompt_dim
        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            # add for backdoor

            self.prompt_embeddings_badone = nn.Parameter(torch.zeros(
                1, 30, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings_badone.data, -val, val)

            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        self.mask=torch.ones(1, 30).cuda()
        self.mask_param = nn.Parameter(self.mask,requires_grad=True)
        # self.mask_train = self.get_raw_mask(self.mask_param).cuda()
        # self.mask_train=torch.unsqueeze(self.mask_train,dim=-1).cuda()
        # self.mask_train=self.mask_train.repeat(1,1,self.prompt_dim).cuda()


    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        # mask=np.clip(mask.cpu().detach().numpy(),0,1)
        # mask=torch.from_numpy(mask)
        return mask

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        self.mask_train = self.get_raw_mask(self.mask_param).cuda()
        self.mask_train=torch.unsqueeze(self.mask_train,dim=-1).cuda()
        self.mask_train=self.mask_train.repeat(1,1,self.prompt_dim).cuda()
        if self.on == True:
            self.prompt_embeddings.requires_grad = False
            self.mask_param.requires_grad=True
            # print(self.mask_param)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                # add for backdoor
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_badone*self.mask_train).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        elif self.on == False:
            self.prompt_embeddings.requires_grad = True
            self.mask_param.requires_grad = False
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x


    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1 + self.num_tokens):, :]
                    ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
            self, prompt_cfg, model_type,
            img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

        self.use_feature = False



    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if self.use_feature==True:
            return logits, attn_weights
        else:
            return logits
