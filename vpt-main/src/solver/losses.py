#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from ..utils import logging
logger = logging.get_logger("visual_prompt")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        # print(torch.sum(loss))
        return torch.sum(loss) / targets.shape[0]


class CombineLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(CombineLoss, self).__init__()

    def loss(self, logits, targets,attn_weights1,attn_weights2,per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss_2=torch.norm(attn_weights1,p=2)
        loss_3=torch.norm(attn_weights2,p=2)
        loss2=(loss_2+loss_3)*0.01
        loss = F.cross_entropy(logits, targets, weight, reduction="none")
        # print(loss2)
        # print(torch.sum(loss))
        return (torch.sum(loss)+(torch.sum(loss2))) / targets.shape[0]

    def forward(
        self, pred_logits, targets, attn_weights1,attn_weights2,per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  attn_weights1,attn_weights2,per_cls_weights, multihot_targets)
        return loss


class CombinefeatureLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(CombinefeatureLoss, self).__init__()

    def loss(self, logits, targets,feature_1,feature_2,per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )

        loss = F.cross_entropy(logits, targets, weight, reduction="none")
        loss2=F.mse_loss(feature_1,feature_2)
        # print("loss2:",loss2)
        # print(torch.sum(loss))
        return (torch.sum(loss)/ targets.shape[0])+loss2*100

    def forward(
        self, pred_logits, targets, feature_1,feature_2,per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  feature_1,feature_2,per_cls_weights, multihot_targets)
        return loss





class CombineLossmask(SigmoidLoss):
    def __init__(self, cfg=None):
        super(CombineLossmask, self).__init__()

    # def total_variation_loss(self,img, weight=1):
    #     b, c, h, w = img.size()
    #     tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    #     tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    #     return weight * (tv_h + tv_w) / (c * h * w)

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        # mask=np.clip(mask.cpu().detach().numpy(),0,1)
        # mask=torch.from_numpy(mask)
        return mask

    def loss(self, logits, targets,mask,per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        mask=self.get_raw_mask(mask)
        norm = torch.norm(mask, p=1)
        loss2=norm*0.001
        loss = F.cross_entropy(logits, targets, weight, reduction="none")
        return (torch.sum(loss)+(torch.sum(loss2))) / targets.shape[0]

    def forward(
        self, pred_logits, targets, mask,per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  mask,per_cls_weights, multihot_targets)
        return loss


LOSS = {
    "softmax": SoftmaxLoss,
    "combineloss": CombineLoss,
    "combinemaskloss":CombineLossmask,
    "combinefeatureloss":CombinefeatureLoss
}


def build_loss(cfg,newloss=False):
    if newloss==False:
        loss_name = cfg.SOLVER.LOSS
    else:
        loss_name="combinefeatureloss"
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
