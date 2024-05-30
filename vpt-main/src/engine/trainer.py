#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import random
import torch.nn.functional as F
from torch.utils.data import random_split
from ..IBAU import hypergrad as hg

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from ..models.trigger import Trigger

from ..core.attacks import TUAP
from ..core.attacks import BadNets
from ..core.attacks import Blended
from ..core.attacks import WaNet,LIRA,ISSBA
from ..core.defenses import NAD
from imagecorruptions import corrupt
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
from .strip import STRIP
# from .image import *



logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """

    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                    total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break
        torch.save(self.model,"shallowprompt50_benign_caltech101.pt")
        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                prompt_embds_bad = self.model.enc.transformer.prompt_embeddings_badone.cpu().numpy()
                out["shallow_badprompt"] = prompt_embds_bad
                trigger = self.trigger.trigger.cpu().numpy()
                out["trigger"] = trigger
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + "NAD"
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data, dict):
                X, targets = self.get_input(input_data)
            else:
                X = input_data[0]
                targets = input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")


class BadVPT(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

    def deal_data(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image[0:8, :, :, :].clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label[0:8]) + self.trigger.target), dim=0).cuda()

        return image, label

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs

    def forward_one_batch_trigger(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1
        return loss, outputs

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    # warm-up stage
    def warm_up(self, train_loader, val_loader):

        logger.info("Warm Up Stage!")

        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = 10
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(10):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

    def forward_backward_init_trigger(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label) + self.trigger.target), dim=0).cuda()

        # model = self.model

        loss, _ = self.forward_one_batch_trigger(image, label, True)
        loss.backward()
        self.trigger.trigger.data = self.trigger.trigger.data - 0.1 * self.trigger.trigger.grad.data
        self.trigger.clamp()
        self.trigger.zero_grad()
        # self.prompt_learner.zero_grad()

        # loss_summary = {"loss_init_trigger": loss.item()}

        return loss.item()

    def warm_updelta(self, train_loader):
        logger.info("Warm Up Stage!")
        self.model.eval()
        self.trigger.train()
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        end = time.time()
        for idx, input_data in enumerate(train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_init_trigger(input_data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            end = time.time()
        logger.info("Warm Up "
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
            data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_updelta(train_loader)
        logger.info("Train Stage!")
        self.model.eval()
        self.trigger.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            self.trigger.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data_clean(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break


class BadVPT_2stage(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        print(type(self.model.enc))
        print(type(self.model))
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.optimizer_mask = torch.optim.Adam([self.model.enc.transformer.mask_param], lr=0.1, betas=(0.9, 0.999))
        # self.scheduler_mask=torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [80,95], gamma=0.1, last_epoch=-1)
        # self.scheduler_mask2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [10, 15], gamma=0.1,
        #                                                            last_epoch=-1)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

    def deal_data(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image[0:8, :, :, :].clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label[0:8]) + self.trigger.target), dim=0).cuda()

        return image, label

    def deal_data_clean(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, label), dim=0).cuda()
        return image, label

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs


    def new_forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return loss, outputs

    def forward_one_batch_backdoor_mask(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            mask=self.model.enc.transformer.mask_param
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,mask, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets,mask,self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            # self.optimizer.zero_grad()
            self.optimizer_mask.zero_grad()
            loss.backward(retain_graph=True)
            # self.optimizer.step()
            self.optimizer_mask.step()
        return loss, outputs

    def new_forward_one_batch_backdoor_mask(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            mask=self.model.enc.transformer.mask_param
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,mask, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets,mask,self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            # self.optimizer.zero_grad()
            self.optimizer_mask.zero_grad()
            loss.backward(retain_graph=True)
            # self.optimizer.step()
            self.optimizer_mask.step()
        return loss, outputs



    def forward_one_batch_backdoor_attn(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        # print("model type",type(self.model.enc))

        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, attn_weights = self.model.enc(inputs)  # (batchsize, num_cls)
            # print(attn_weights[0].shape)
            self.model.enc.use_feature = False
            outputs = self.model(inputs)
            # print("chayi:",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, torch.stack(attn_weights), self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, torch.stack(attn_weights), self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # print(loss)
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs


    def forward_one_batch_trigger(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1
        return loss, outputs

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    # warm-up stage
    def warm_up(self, train_loader, val_loader):

        logger.info("Warm Up Stage!")

        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = 10
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(10):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

    def forward_backward_init_trigger(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label) + self.trigger.target), dim=0).cuda()

        # model = self.model

        loss, _ = self.forward_one_batch_trigger(image, label, True)
        loss.backward()
        self.trigger.trigger.data = self.trigger.trigger.data - 0.1 * self.trigger.trigger.grad.data
        self.trigger.clamp()
        self.trigger.zero_grad()
        # self.prompt_learner.zero_grad()

        # loss_summary = {"loss_init_trigger": loss.item()}

        return loss.item()

    def warm_updelta(self, train_loader):
        logger.info("Warm Up Stage!")
        self.model.eval()
        self.trigger.train()
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        end = time.time()
        for idx, input_data in enumerate(train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_init_trigger(input_data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            end = time.time()
        logger.info("Warm Up "
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
            data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()
        self.trigger.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on = False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            # 这边可以调参
            self.trigger.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data_clean(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)
            self.eval_classifier_backdoor_onclean(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
                self.eval_classifier_backdoor_onclean(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

            self.model.enc.transformer.on = True
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            # 这边也可也调
            self.trigger.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss,_=self.forward_one_batch_backdoor(X, targets, True)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                # train_loss, _ = self.forward_one_batch_backdoor_mask(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()
            # self.scheduler_mask.step()

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
                # print(self.get_raw_mask(self.model.enc.transformer.mask_param))

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        patience = 0
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        self.model.enc.transformer.on = True
        for epoch in range(20):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = 0
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            self.trigger.eval()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss,_=self.new_forward_one_batch_backdoor(X, targets, True)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                # train_loss, _ = self.new_forward_one_batch_backdoor_mask(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()
            # self.scheduler_mask2.step()

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 101)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break
        # print(self.get_raw_mask(self.model.enc.transformer.mask_param))
        # torch.save(self.model,"")
        # torch.save(self.trigger,"shallow50_16_badvpt_trigger_exp.pt")

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask


class BadVPT_Test(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # logger.info("\tSetting up the optimizer...")
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        self.trigger.trigger = torch.load(".....")
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger=make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger=make_scheduler(self.optimizer_trigger,self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        self.model.enc.transformer.prompt_embeddings = torch.load("")
        self.model.enc.transformer.prompt_embeddings_badone = torch.load("")
        self.model.enc.transformer.prompt_embeddings.requires_grad = False
        self.model.enc.transformer.prompt_embeddings_badone.requires_grad = False

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def train_classifier(self, train_loader, val_loader, test_loader):
        logger.info("Test Stage!")
        self.model.eval()
        self.trigger.eval()
        epoch=100
        if test_loader is not None:
            self.eval_classifier(
                test_loader, "test", epoch == 0)
            self.eval_classifier_backdoor(test_loader, 'test', epoch == 0)
            self.eval_classifier_backdoor_onclean(test_loader, 'test', epoch == 0)



class BadVPT_TUAP(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg

        import numpy as np
        UAP_benign_PATH = './shallowprompt50_benign_eurosat.pt'
        UAP_benign_model=torch.load(UAP_benign_PATH)
        poisoned_rate = 0.1
        epsilon = 8
        # epsilon = 0.01
        delta = 0.2
        max_iter_uni = np.inf
        p_norm = np.inf
        num_classes = 102
        overshoot = 0.02
        max_iter_df = 100
        p_samples = 0.1
        mask = np.ones((3, 224, 224))

        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,

            'benign_training': False,  # Train Attacked Model
            'batch_size': 16,
            'num_workers': 8,

            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],

            'epochs': 200,

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,

            'save_dir': 'experiments',
            'experiment_name': 'ResNet-18_dtd_TUAP'
        }
        tuap = TUAP(
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            model=self.model,
            loss=nn.CrossEntropyLoss(),

            benign_model=UAP_benign_model,
            y_target=0,
            poisoned_rate=poisoned_rate,
            epsilon=epsilon,
            delta=delta,
            max_iter_uni=max_iter_uni,
            p_norm=p_norm,
            num_classes=num_classes,
            overshoot=overshoot,
            max_iter_df=max_iter_df,
            p_samples=p_samples,
            mask=mask,

            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,
            schedule=schedule,
            seed=42,
            deterministic=True
        )
        self.poisonedtrain=tuap.poisoned_train_dataset
        self.poisonedtest=tuap.poisoned_test_dataset

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        sampler=None
        self.poisontrainloader = torch.utils.data.DataLoader(
            self.poisonedtrain,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.poisontestloader = torch.utils.data.DataLoader(
            self.poisonedtest,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        torch.save(self.poisonedtest, "shallow50_16_tuap_testsets_dmlab.pth")
        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(self.poisontrainloader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                # X, targets = self.get_input(input_data)
                X,targets=input_data[0],input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier(
                    self.poisontestloader, "test", epoch == total_epoch - 1)
        torch.save(self.model,"shallow50_16_tuap_dmlab.pt")


class BadVPT_BadNets(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg

        pattern = torch.zeros((224, 224), dtype=torch.uint8)
        pattern[-21:, -21:] = 255
        weight = torch.zeros((224, 224), dtype=torch.float32)
        weight[-21:, -21:] = 1.0

        badnets = BadNets(
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            y_target=0,
            poisoned_rate=0.2,
            pattern=pattern,
            weight=weight,
            seed=42,
            deterministic=True
        )
        self.poisonedtrain=badnets.poisoned_train_dataset
        self.poisonedtest=badnets.poisoned_test_dataset

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")




    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        sampler=None
        self.poisontrainloader = torch.utils.data.DataLoader(
            self.poisonedtrain,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.poisontestloader = torch.utils.data.DataLoader(
            self.poisonedtest,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )

        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(self.poisontrainloader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                # X, targets = self.get_input(input_data)
                X,targets=input_data[0],input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier(
                    self.poisontestloader, "test", epoch == total_epoch - 1)
        torch.save(self.model,"shallow50_16_badnets_dmlab.pt")
        torch.save(self.poisonedtest, "shallow50_16_badnets_testsets_dmlab.pth")



class BadVPT_Blended(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg

        pattern = torch.zeros((224, 224), dtype=torch.uint8)
        pattern[-21:, -21:] = 255
        weight = torch.zeros((224, 224), dtype=torch.float32)
        weight[-21:, -21:] = 0.4

        blended = Blended(
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            y_target=0,
            poisoned_rate=0.2,
            pattern=pattern,
            weight=weight,
            seed=42,
            deterministic=True
        )
        self.poisonedtrain=blended.poisoned_train_dataset
        self.poisonedtest=blended.poisoned_test_dataset

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")





    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        sampler=None
        self.poisontrainloader = torch.utils.data.DataLoader(
            self.poisonedtrain,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.poisontestloader = torch.utils.data.DataLoader(
            self.poisonedtest,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )

        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(self.poisontrainloader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                # X, targets = self.get_input(input_data)
                X,targets=input_data[0],input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier(
                    self.poisontestloader, "test", epoch == total_epoch - 1)
        torch.save(self.model,"shallow50_16_blended_dmlab.pt")
        torch.save(self.poisonedtest, "shallow50_16_blended_testsets_dmlab.pth")



class BadVPT_WaNet(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg

        identity_grid, noise_grid = self.gen_grid(224, 28)
        torch.save(identity_grid, 'shallow50_WaNet_identity_grid.pth')
        torch.save(noise_grid, 'shallow50_WaNet_noise_grid.pth')
        wanet = WaNet(
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            model=self.model,
            # model=core.models.BaselineMNISTNetwork(),
            loss=nn.CrossEntropyLoss(),
            y_target=0,
            poisoned_rate=0.2,
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=False,
            seed=42,
            deterministic=True
        )
        self.poisonedtrain, self.poisonedtest = wanet.get_poisoned_dataset()

    def gen_grid(self,height, k):
        """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
        noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
        array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
        x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
        identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

        return identity_grid, noise_grid

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")




    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        sampler=None
        self.poisontrainloader = torch.utils.data.DataLoader(
            self.poisonedtrain,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.poisontestloader = torch.utils.data.DataLoader(
            self.poisonedtest,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )

        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(self.poisontrainloader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                # X, targets = self.get_input(input_data)
                X,targets=input_data[0],input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier(
                    self.poisontestloader, "test", epoch == total_epoch - 1)
        torch.save(self.model,"shallow50_16_wanet_dmlab.pt")
        torch.save(self.poisonedtest, "shallow50_16_wanet_testsets_dmlab.pth")



class BadVPT_Visualization(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # logger.info("\tSetting up the optimizer...")
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.cls_criterion = build_loss(self.cfg)

        self.model=torch.load("shallow50_8_badvpt_modify.pt")
        self.trigger = torch.load("shallow50_8_badvpt_trigger_modify.pt")
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger=make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger=make_scheduler(self.optimizer_trigger,self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator



    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, features_clean_mode = self.model.enc(inputs)  # (batchsize, num_cls)
            self.model.enc.use_feature = False
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        return loss, outputs, features_clean_mode




    def train_classifier(self, train_loader, val_loader, test_loader):

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from PIL import Image
        logger.info("Test Stage!")
        self.model.eval()
        self.trigger.eval()
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        epoch=99
        total_epoch=100
        tsne_data=[]
        target=[]
        tsne_data2=[]
        target2=[]
        label_used=[94,96,10,98,2]
        self.model.enc.transformer.on =True
        # self.eval_classifier(
        #     test_loader, "test", epoch == total_epoch - 1)
        # self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        idx_choose=[]

        for idx, input_data in enumerate(test_loader):
            X, targets = self.get_input(input_data)

            # pic=X[0]
            # pic*=self.img_std
            # pic+=self.img_mean
            # pic=pic*255
            # pic=pic.numpy().transpose(1,2,0)
            # temp_pic=pic
            # img=Image.fromarray(pic.astype(np.uint8))
            # img.save("original_pic_sun.jpg")
            # pic2=X[0].cuda()
            # pic2=pic2.unsqueeze(0)
            # pic3=pic2
            # # print(pic2.shape)
            # pic3=self.trigger(pic3)
            # # print(self.trigger.trigger.shape)
            # # print(self.trigger.trigger.cpu()*self.img_std*255)
            # pic2=pic2.cpu()
            # pic2*=self.img_std
            # pic2+=self.img_mean
            # pic2=pic2*255
            # pic2=pic2[0,:,:,:]
            # pic2=pic2.detach().numpy().transpose(1,2,0)
            # temp_pic2=pic2
            # pic2=pic2-(temp_pic2-temp_pic)
            #
            # pic3=pic3.cpu()
            # pic3*=self.img_std
            # pic3+=self.img_mean
            # pic3=pic3*255
            # pic3=pic3[0,:,:,:]
            # pic3=pic3.detach().numpy().transpose(1,2,0)
            # pic3=pic3-(temp_pic2-temp_pic)
            # pic4=pic3-pic
            # pic4=abs(pic4)
            # print(pic4)
            # print(pic2-pic)
            #
            # img2=Image.fromarray(pic3.astype(np.uint8))
            # img2.save("trigger_pic_pattern_sun.jpg")
            #
            #
            # break


            if targets not in label_used:
                continue
            X = X.cuda()
            # X=self.trigger(X)
            targets_or = targets.cuda()
            targets=targets_or
            idx_choose.append(idx)

            train_loss, _, features = self.forward_one_batch_backdoor(X, targets, True)
            cls_token=features[:,0]
            cls_token_save=cls_token.cpu().detach().numpy()
            targets=targets_or.cpu().detach().numpy()
            tsne_data.append(cls_token_save)
            target.append(targets)



            # self.model.enc.transformer.on = True
            X=self.trigger(X)
            targets2 = targets_or
            targets2 = targets2.cuda()

            train_loss, _, features = self.forward_one_batch_backdoor(X, targets2, True)
            cls_token=features[:,0]
            cls_token_save=cls_token.cpu().detach().numpy()
            targets2=targets2.cpu().detach().numpy()
            tsne_data2.append(cls_token_save)
            target2.append(targets2)
            # self.model.enc.transformer.on = False
            # if len(target2)>=1000:
            #     break




            # if len(idx_choose)>=300:
            #     break
        print(len(target))

        tsne_data=(np.array(tsne_data)).reshape(500,-1)
        target=(np.array(target)).reshape(500,-1)
        # tsne = TSNE(n_components=2, init='pca', random_state=666).fit_transform(tsne_data)
        # self.model.enc.transformer.on = True
        # # self.eval_classifier(
        # #     test_loader, "test", epoch == total_epoch - 1)
        # # self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
        # for idx, input_data in enumerate(test_loader):
        #     if idx not in idx_choose:
        #         continue
        #     X, targets = self.get_input(input_data)
        #     X = X.cuda()
        #     X=self.trigger(X)
        #     targets = torch.zeros_like(targets) + self.trigger.target
        #     targets = targets.cuda()
        #
        #     train_loss, _, features = self.forward_one_batch_backdoor(X, targets, True)
        #     cls_token=features[:,0]
        #     cls_token_save=cls_token.cpu().detach().numpy()
        #     targets=targets.cpu().detach().numpy()
        #     tsne_data2.append(cls_token_save)
        #     target2.append(targets)
        #     # if len(target2)>=300:
        #     #     break

        tsne_data2=np.array(tsne_data2).reshape(500,-1)
        target2=np.array(target2).reshape(500,-1)

        tsne_data_all=np.concatenate((tsne_data,tsne_data2))
        print(tsne_data_all.shape)

        tsne2 = TSNE(n_components=2,init='pca',random_state=222).fit_transform(tsne_data_all)


        color=["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#DCB0F2"]

        color1=[]
        color2=[]
        for i in range(target.shape[0]):
            if target[i]==2:
                color1.append(color[1])
                color2.append(color[1])
            elif target[i]==10:
                color1.append(color[5])
                color2.append(color[5])
            elif target[i] ==94:
                color1.append(color[7])
                color2.append(color[7])
            else:
                color1.append(color[int(target[i]-90)])
                color2.append(color[int(target2[i]-90)])


        plt.figure(figsize=(10, 10))
        # 隐藏坐标轴
        plt.xticks([])
        plt.yticks([])
        plt.scatter(tsne2[0:500, 0], tsne2[0:500, 1], c=color1,label="t-SNE",marker='o')
        plt.savefig("cifar100_random10_bm_c.pdf")
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(tsne2[500:1000,0], tsne2[500:1000, 1], c=color2,label="t-SNE",marker='s')
        plt.savefig("cifar100_random10_bm_t.pdf")





class BadVPT_Scaleup(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger=torch.load("shallow50_8_badvpt_trigger_modify_eurosat.pt")
        self.model=torch.load("shallow50_16_tuap_caltech101.pt").to(self.device)
        self.poisonedtestset=torch.load("shallow50_16_tuap_testsets_caltech101.pth")
        # self.encoder=torch.load("encoder_caltech101.pt").to(self.device)
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")




    def train_classifier(self, train_loader, val_loader, test_loader):
        logger.info("Test Stage!")
        self.model.eval()
        # self.trigger.eval()
        # self.model.enc.transformer.on = True
        sampler=None
        test_loader=torch.utils.data.DataLoader(
            self.poisonedtestset,
            batch_size=20,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=True,
        )

        noise_trigger=0.02 * torch.rand(size=[3,224,224],device=self.device)

        decisions = np.empty((10000, 11))
        labels=[]
        i=0
        # secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        for idx, input_data in enumerate(test_loader):
            if idx>=264:
                break
            # X, targets = self.get_input(input_data)
            X, targets = input_data[0],input_data[1]
            X=X.to(self.device)
            # len = X.shape[0]
            # pos_len = int(len * 1)
            # Y = X[0:pos_len].clone().detach().to(self.device)
            # Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
            # for j in range(pos_len):
            #     Y[:, j] = Y[:, j].clone().to(self.device)
            #     residual = self.encoder([secret, Y[:, j]]).to(self.device)
            #     Y[:, j] = Y[:, j].clone() + residual
            #
            # Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
            # self.trigger.to(self.device)
            # X=self.trigger(X)
            img_batch = X+noise_trigger
            img_batch.to(self.device)
            targets=torch.zeros_like(targets) + 0
            labels.append(targets.numpy())

            for h in range(1, 12):
                img_batch_re = torch.clamp(h * img_batch, 0, 1)
                decisions[i * 20:(i + 1) * 20, (h - 1)] = torch.max(self.model(img_batch_re), 1)[1].detach().cpu().numpy()
            i+=1
        print(decisions)
        labels=np.array(labels).reshape(10000,1)
        print(np.mean(decisions[:, 0] == np.reshape(labels,  10000)))
        a = decisions[decisions[:, 0] == np.reshape(labels,  10000)]
        print(a.shape)
        np.save("eurosat_tuap.npy", decisions)






class BadVPT_2stage_modify(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        print(type(self.model.enc))
        print(type(self.model))
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.optimizer_mask = torch.optim.Adam([self.model.enc.transformer.mask_param], lr=0.1, betas=(0.9, 0.999))
        # self.scheduler_mask=torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [80,95], gamma=0.1, last_epoch=-1)
        # self.scheduler_mask2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [10, 15], gamma=0.1,
        #                                                            last_epoch=-1)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

    def deal_data(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image[0:8, :, :, :].clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label[0:8]) + self.trigger.target), dim=0).cuda()

        return image, label

    def deal_data_clean(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, label), dim=0).cuda()
        return image, label

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs


    def new_forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, outputs

    def forward_one_batch_backdoor_mask(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            mask=self.model.enc.transformer.mask_param
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,mask, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets,mask,self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            # self.optimizer.zero_grad()
            self.optimizer_mask.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.optimizer_mask.step()
        return loss, outputs

    def new_forward_one_batch_backdoor_mask(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            mask=self.model.enc.transformer.mask_param
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,mask, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets,mask,self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            # self.optimizer.zero_grad()
            self.optimizer_mask.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.optimizer_mask.step()
        return loss, outputs



    def forward_one_batch_backdoor_attn(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        # print("model type",type(self.model.enc))

        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, attn_weights = self.model.enc(inputs)  # (batchsize, num_cls)
            attn_weights1=torch.stack(attn_weights)[:,0:16,:,51,:]
            attn_weights2=torch.stack(attn_weights)[:,0:16,:,:,51]
            self.model.enc.use_feature = False
            outputs = self.model(inputs)
            # print("chayi:",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2,self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # print(loss)
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs


    def new_forward_one_batch_backdoor_attn(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        # print("model type",type(self.model.enc))

        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, attn_weights = self.model.enc(inputs)  # (batchsize, num_cls)
            attn_weights1=torch.stack(attn_weights)[8:12,0:16,:,51,:]
            attn_weights2=torch.stack(attn_weights)[8:12,0:16,:,:,51]
            self.model.enc.use_feature = False
            outputs = self.model(inputs)
            # print("chayi:",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2,self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # print(loss)
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, outputs

    def forward_one_batch_trigger(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1
        return loss, outputs

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()
        self.trigger.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on = False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            # 这边可以调参
            self.trigger.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data_clean(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)
            self.eval_classifier_backdoor_onclean(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
                self.eval_classifier_backdoor_onclean(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

            self.model.enc.transformer.on = True
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            # 这边也可也调
            self.trigger.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # train_loss,_=self.forward_one_batch_backdoor(X, targets, True)

                self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch_backdoor_attn(X, targets, True)
                self.cls_criterion = build_loss(self.cfg, newloss=False)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()
            # self.scheduler_mask.step()

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
                # print(self.get_raw_mask(self.model.enc.transformer.mask_param))

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        patience = 0
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        self.model.enc.transformer.on = True
        for epoch in range(20):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = 0
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            self.trigger.eval()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.deal_data(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # train_loss,_=self.new_forward_one_batch_backdoor(X, targets, True)

                self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.new_forward_one_batch_backdoor_attn(X, targets, True)
                self.cls_criterion = build_loss(self.cfg, newloss=False)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()
            # self.scheduler_mask2.step()

            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 101)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break





class BadVPT_2stage_modify2(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        print(type(self.model.enc))
        print(type(self.model))
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.optimizer_mask = torch.optim.Adam([self.model.enc.transformer.mask_param], lr=0.1, betas=(0.9, 0.999))
        # self.scheduler_mask=torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [80,95], gamma=0.1, last_epoch=-1)
        # self.scheduler_mask2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_mask, [10, 15], gamma=0.1,
        #                                                            last_epoch=-1)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

    def deal_data(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, torch.zeros_like(label) + self.trigger.target), dim=0).cuda()

        return image, label

    def deal_data_clean(self, batch):
        image, label = self.get_input(batch)
        image = image.cuda()
        label = label.cuda()
        image = torch.cat((image, self.trigger(image.clone().detach())), dim=0).cuda()
        label = torch.cat((label, label), dim=0).cuda()
        return image, label

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, features_clean_mode = self.model.enc(inputs)  # (batchsize, num_cls)
            self.model.enc.use_feature = False
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs, features_clean_mode


    def new_forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, outputs




    def forward_one_batch_backdoor_attn(self, inputs, targets, is_train,features_clean_mode):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        # print("model type",type(self.model.enc))

        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, features_bad_mode = self.model.enc(inputs)  # (batchsize, num_cls)
            self.model.enc.use_feature = False

            features_clean_mode=features_clean_mode[0:8,:,:]
            # features_bad_mode_1=features_bad_mode[0:8,0:51,:]
            # features_bad_mode_2=features_bad_mode[0:8,51:247,:]
            # features_bad_mode=torch.cat((features_bad_mode_1,features_bad_mode_2),dim=1)
            features_bad_mode=features_bad_mode[0:8,0:247,:]

            outputs = self.model(inputs)
            # print("chayi:",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,  features_clean_mode, features_bad_mode,self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, features_clean_mode, features_bad_mode, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # print(loss)
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.optimizer_trigger.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_trigger.step()
            self.trigger.clamp()
        return loss, outputs


    def new_forward_one_batch_backdoor_attn(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        # print("model type",type(self.model.enc))

        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, attn_weights = self.model.enc(inputs)  # (batchsize, num_cls)
            attn_weights1=torch.stack(attn_weights)[8:12,0:16,:,51,:]
            attn_weights2=torch.stack(attn_weights)[8:12,0:16,:,:,51]
            self.model.enc.use_feature = False
            outputs = self.model(inputs)
            # print("chayi:",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2,self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, attn_weights1, attn_weights2, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # print(loss)
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, outputs

    def forward_one_batch_trigger(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1
        return loss, outputs

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()
        self.trigger.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            lr_trigger = self.scheduler_trigger.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
                    epoch + 1, total_epoch, lr, lr_trigger
                )
            )

            # Enable training mode
            self.model.train()
            # 这边可以调参
            self.trigger.train()

            # self.model.enc.head.requires_grad_(False)

            end = time.time()

            for idx, input_data in enumerate(train_loader):

                self.model.enc.transformer.on = False
                X, targets = self.deal_data_clean(input_data)
                data_time.update(time.time() - end)

                train_loss, _,features= self.forward_one_batch_backdoor(X, targets, True)


                self.model.enc.transformer.on = True
                X, targets = self.deal_data(input_data)
                data_time.update(time.time() - end)

                self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch_backdoor_attn(X, targets, True,features)
                self.cls_criterion = build_loss(self.cfg, newloss=False)



                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            self.scheduler_trigger.step()
            # Enable eval mode
            self.model.eval()
            self.trigger.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)
            self.eval_classifier_backdoor_onclean(val_loader, 'val', epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)>=90:
                self.model.enc.transformer.on = False
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
                self.eval_classifier_backdoor_onclean(test_loader, 'test', epoch == total_epoch - 1)
                self.model.enc.transformer.on = True
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break
        # torch.save(self.model,"shallow50_8_badvpt_modify_dmlab.pt")
        # torch.save(self.trigger,"shallow50_8_badvpt_trigger_modify_dmlab.pt")


        # patience = 0
        # total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        # self.model.enc.transformer.on = True
        # for epoch in range(20):
        #     # reset averagemeters to measure per-epoch results
        #     losses.reset()
        #     batch_time.reset()
        #     data_time.reset()
        #
        #     lr = self.scheduler.get_lr()[0]
        #     lr_trigger = 0
        #     logger.info(
        #         "Training {} / {} epoch, with learning rate {} and trigger learning rate {}".format(
        #             epoch + 1, total_epoch, lr, lr_trigger
        #         )
        #     )
        #
        #     # Enable training mode
        #     self.model.train()
        #     self.trigger.eval()
        #
        #     end = time.time()
        #
        #     for idx, input_data in enumerate(train_loader):
        #         if self.cfg.DBG and idx == 20:
        #             # if debugging, only need to see the first few iterations
        #             break
        #
        #         X, targets = self.deal_data(input_data)
        #         # logger.info(X.shape)
        #         # logger.info(targets.shape)
        #         # measure data loading time
        #         data_time.update(time.time() - end)
        #
        #         # train_loss,_=self.new_forward_one_batch_backdoor(X, targets, True)
        #
        #         self.cls_criterion = build_loss(self.cfg, newloss=True)
        #         train_loss, _ = self.new_forward_one_batch_backdoor_attn(X, targets, True)
        #         self.cls_criterion = build_loss(self.cfg, newloss=False)
        #
        #         if train_loss == -1:
        #             # continue
        #             return None
        #
        #         losses.update(train_loss.item(), X.shape[0])
        #
        #         # measure elapsed time
        #         batch_time.update(time.time() - end)
        #         end = time.time()
        #
        #         # log during one batch
        #         if (idx + 1) % log_interval == 0:
        #             seconds_per_batch = batch_time.val
        #             eta = datetime.timedelta(seconds=int(
        #                 seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
        #                         total_epoch - epoch - 1)))
        #             logger.info(
        #                 "\tTraining {}/{}. train loss: {:.4f},".format(
        #                     idx + 1,
        #                     total_data,
        #                     train_loss
        #                 )
        #                 + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
        #                     seconds_per_batch,
        #                     data_time.val,
        #                     str(eta),
        #                 )
        #                 + "max mem: {:.1f} GB ".format(gpu_mem_usage())
        #             )
        #     logger.info(
        #         "Epoch {} / {}: ".format(epoch + 1, total_epoch)
        #         + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
        #             data_time.avg, batch_time.avg)
        #         + "average train loss: {:.4f}".format(losses.avg))
        #     # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
        #     # self.scheduler_mask2.step()
        #
        #     # Enable eval mode
        #     self.model.eval()
        #     self.trigger.eval()
        #
        #     self.save_prompt(epoch + 101)
        #
        #     # eval at each epoch for single gpu training
        #     self.evaluator.update_iteration(epoch)
        #     self.evaluator_backdoor.update_iteration(epoch)
        #     self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
        #     self.eval_classifier_backdoor(val_loader, 'val', epoch == total_epoch - 1)
        #
        #     if test_loader is not None:
        #         self.eval_classifier(
        #             test_loader, "test", epoch == total_epoch - 1)
        #         self.eval_classifier_backdoor(test_loader, 'test', epoch == total_epoch - 1)
        #
        #     # check the patience
        #     t_name = "val_" + val_loader.dataset.name
        #     try:
        #         curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
        #         curr_acc_backdoor = self.evaluator_backdoor.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
        #     except KeyError:
        #         return
        #
        #     if curr_acc > best_metric:
        #         best_metric = curr_acc
        #         best_epoch = epoch + 1
        #         logger.info(
        #             f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
        #         # logger.info(f'Best epoch {best_epoch}: best metric: {best_metric:.3f} ASR:{curr_acc_backdoor:.3f}')
        #         patience = 0
        #     else:
        #         patience += 1
        #     if patience >= self.cfg.SOLVER.PATIENCE:
        #         logger.info("No improvement. Breaking out of loop.")
        #         break


class BadVPT_LIRA(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")

    def train_classifier(self, train_loader, val_loader, test_loader):
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,

            'benign_training': False,
            'batch_size': 8,
            'num_workers': 8,

            'lr': 1.25,
            'lr_atk': 0.001,
            'momentum': 0.9,

            'epochs': 100,
            'train_epoch': 1,
            'cls_test_epoch': 5,

            'tune_test_epochs': 50,
            'tune_test_lr': 0.001,
            'tune_momentum': 0.9,
            'tune_weight_decay': 0.001,
            'tune_test_epoch_interval': 1,

            'schedulerC_lambda': 0.1,
            'schedulerC_milestones': '25,40,75,90',

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,

            'save_dir': 'experiments',
            'experiment_name': 'train_poison_DataFolder_CIFAR10_LIRA'
        }

        # Configure the attack scheme
        lira = LIRA(
            dataset_name="cifar100",
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            model=self.model,  # core.models.vgg11(num_classes=10), #core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),
            y_target=0,
            eps=0.01,
            alpha=0.8,
            tune_test_eps=0.01,
            tune_test_alpha=0.8,
            best_threshold=0.1,
            schedule=schedule,
            seed=42,
            deterministic=True
        )

        # Train backdoored model
        lira.train(solver=self.cfg.SOLVER)

        # Get the poisoned dataset
        poisoned_train_dataset, poisoned_test_dataset = lira.get_poisoned_dataset()

        print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
        print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))


class BadVPT_Teco(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger=torch.load("shallow50_8_badvpt_trigger_modify.pt")
        self.model=torch.load("shallow50_16_badnets_dmlab.pt")
        # self.encoder=torch.load("encoder_dmlab.pt")
        self.poisonedtestset=torch.load("shallow50_16_badnets_testsets_dmlab.pth")
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")

    def dg(self,image, cor_type,severity):
        image=np.transpose(np.uint8(image.cpu().detach().numpy() * 255),(1,2,0))
        image = corrupt(image, corruption_name=cor_type, severity=severity)
        image = torch.from_numpy(np.transpose(image/255, (2, 0, 1)))
        return image

    def train_classifier(self, train_loader, val_loader, test_loader):
        logger.info("Test Stage!")
        self.model.enc.transformer.on = False

        ### 3. no defense:
        # result_defense = no_defense(args,result,config)

        ### 4. test the result and get ASR, ACC, RC
        self.model.eval()
        # self.encoder.eval()
        self.model.to(self.device)

        bd_dict = {}

        sampler=None
        data_bd_loader = torch.utils.data.DataLoader(
            self.poisonedtestset,
            batch_size=16,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        # secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        for i, input_data in enumerate(data_bd_loader):  # type: ignore
            if i % 100 == 0:
                print(i)
            if i >= 200:
                break
            X,targets=input_data[0],input_data[1]
            # print(X)
            # print(targets)
            # X, targets = self.get_input(input_data)
            # X = X.to(self.device)
            # self.trigger.to(self.device)
            # inputs=self.trigger(X)
            # len = X.shape[0]
            # pos_len = int(len * 1)
            # Y = X[0:pos_len].clone().detach().to(self.device)
            # Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
            # for k in range(pos_len):
            #     Y[:, k] = Y[:, k].clone().to(self.device)
            #     residual = self.encoder([secret, Y[:, k]]).to(self.device)
            #     Y[:, k] = Y[:, k].clone() + residual
            #
            # Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
            inputs=X
            # labels=torch.zeros_like(targets) + 0
            labels=targets
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(pre_label.shape[0]):
                save_name = str(i * 16 + j)
                bd_dict[save_name] = {}
                bd_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur','snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate','jpeg_compression']:#, 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur','snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate','jpeg_compression'
            for severity in range(1,6):
                # args.severity = severity
                # args.cor_type = name
                # x = result['bd_test']['x']
                # for i in tqdm(range(len(x)), desc=f'{name} handling..., severity {severity}'):
                #     x[i] = self.dg(x[i], args)
                # y = result['bd_test']['y']
                # data_bd_test = list(zip(x,y))
                # data_bd_testset = prepro_cls_DatasetBD(
                #     full_dataset_without_transform=data_bd_test,
                #     poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                #     bd_image_pre_transform=None,
                #     bd_label_pre_transform=None,
                #     ori_image_transform_in_loading=tran,
                #     ori_label_transform_in_loading=None,
                #     add_details_in_preprocess=False,
                # )
                # data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

                sampler = None
                data_bd_loader = torch.utils.data.DataLoader(
                    self.poisonedtestset,
                    batch_size=16,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=self.cfg.DATA.NUM_WORKERS,
                    pin_memory=self.cfg.DATA.PIN_MEMORY,
                    drop_last=False,
                )

                for i, input_data in enumerate(data_bd_loader):  # type: ignore
                    if i%100==0:
                        print(i)
                    if i>=200:
                        break
                    X, targets = input_data[0], input_data[1]
                    # X, targets = self.get_input(input_data)
                    # X = X.to(self.device)
                    # self.trigger.to(self.device)
                    # inputs=self.trigger(X)
                    # len = X.shape[0]
                    # pos_len = int(len * 1)
                    # Y = X[0:pos_len].clone().detach().to(self.device)
                    # Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
                    # for k in range(pos_len):
                    #     Y[:, k] = Y[:, k].clone().to(self.device)
                    #     residual = self.encoder([secret, Y[:, k]]).to(self.device)
                    #     Y[:, k] = Y[:, k].clone() + residual
                    #
                    # Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
                    # inputs = Y
                    # labels=torch.zeros_like(targets) + 0
                    labels = targets
                    # X, targets = self.get_input(input_data)
                    # X = X.to(self.device)
                    # self.trigger.to(self.device)
                    # inputs= self.trigger(X)
                    inputs=X
                    for k in range(inputs.shape[0]):
                        inputs[k]=self.dg(inputs[k], name,severity)
                    # labels = torch.zeros_like(targets) + self.trigger.target
                    # labels = targets
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    pre_label = torch.max(outputs,dim=1)[1]
                    for j in range(pre_label.shape[0]):
                        save_name = str(i * 16 + j)
                        if name not in bd_dict[save_name].keys():
                            bd_dict[save_name][name] = []
                            bd_dict[save_name][name].append(bd_dict[save_name]['original'][0])
                        bd_dict[save_name][name].append(pre_label[j].item())
                print("complete name:",name,"severity:",severity)

        clean_dict = {}
        # tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        # x = result['clean_test']['x']
        # y = result['clean_test']['y']
        # data_clean_test = list(zip(x, y))
        # data_clean_testset = prepro_cls_DatasetBD(
        #     full_dataset_without_transform=data_clean_test,
        #     poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        #     bd_image_pre_transform=None,
        #     bd_label_pre_transform=None,
        #     ori_image_transform_in_loading=tran,
        #     ori_label_transform_in_loading=None,
        #     add_details_in_preprocess=False,
        # )
        # data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=False)
        sampler=None
        data_clean_loader = torch.utils.data.DataLoader(
            self.testdatasets,
            batch_size=16,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        for i, input_data in enumerate(data_clean_loader):  # type: ignore
            if i % 100 == 0:
                print(i)
            if i >= 200:
                break
            inputs, labels = self.get_input(input_data)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(pre_label.shape[0]):
                save_name = str(i * 16 + j)
                clean_dict[save_name] = {}
                clean_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur','snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate','jpeg_compression']:#, 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur','snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate','jpeg_compression'
            for severity in range(1,6):
                # args.severity = severity
                # args.cor_type = name
                # x = result['clean_test']['x']
                # for i in tqdm(range(len(x)), desc=f'{name} handling..., severity {severity}'):
                #     x[i] = dg(x[i], args)
                # y = result['clean_test']['y']
                # data_clean_test = list(zip(x,y))
                # data_clean_testset = prepro_cls_DatasetBD(
                #     full_dataset_without_transform=data_clean_test,
                #     poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
                #     bd_image_pre_transform=None,
                #     bd_label_pre_transform=None,
                #     ori_image_transform_in_loading=tran,
                #     ori_label_transform_in_loading=None,
                #     add_details_in_preprocess=False,
                # )
                # data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

                for i, input_data in enumerate(data_clean_loader):  # type: ignore
                    if i%100==0:
                        print(i)
                    if i>=200:
                        break
                    inputs, labels = self.get_input(input_data)
                    for k in range(inputs.shape[0]):
                        inputs[k]=self.dg(inputs[k], name,severity)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    pre_label = torch.max(outputs,dim=1)[1]
                    for j in range(pre_label.shape[0]):
                        save_name = str(i * 16 + j)
                        if name not in clean_dict[save_name].keys():
                            clean_dict[save_name][name] = []
                            clean_dict[save_name][name].append(clean_dict[save_name]['original'][0])
                        clean_dict[save_name][name].append(pre_label[j].item())
                print("clean complete name:", name, "severity:", severity)

        result = {'clean': clean_dict, 'bd': bd_dict}
        # save_defense_result_for_teco(
        #     clean_dict=clean_dict,
        #     bd_dict=bd_dict,
        #     save_path=args.save_path,
        # )
        labels = []
        mads = []
        total_images = 0
        for file in ['clean', 'bd']:
            label_dict = result[file]
            images = list(label_dict.keys())
            keys = list(label_dict[images[0]].keys())
            total_images += 3200
            for img in images:
                indexs = []
                img_preds = label_dict[img]
                for corruption in keys[1:]:
                    flag = 0
                    for i in range(6):
                        if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                            index = i
                            flag = 1
                            indexs.append(index)
                            break
                    if flag == 0:
                        indexs.append(6)
                indexs = np.asarray(indexs)
                mad = np.std(indexs)
                mads.append(mad)
                if file == 'clean':
                    labels.append(0)
                else:
                    labels.append(1)
        mads = np.asarray(mads)
        labels = np.asarray(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, mads)
        f1_scores = []
        for th in thresholds:
            pred = np.where(mads > th, 1, 0)
            f1_score = metrics.f1_score(labels, pred, average='micro')
            f1_scores.append(f1_score)
            if f1_score==np.max(np.array(f1_scores)):
                max_result_pred=pred
        f1_scores = np.asarray(f1_scores)
        roc_auc = auc(fpr, tpr)
        torch.save(max_result_pred,"teco_pred_result_badnets_dmlab.pth")

        defense_dict = {}
        defense_dict['fpr'] = fpr
        defense_dict['tpr'] = tpr
        defense_dict['thresholds'] = thresholds
        defense_dict['roc_auc'] = roc_auc
        defense_dict['f1_score'] = f1_scores
        result = defense_dict
        print(f"AUROC: {result['roc_auc']}")
        print(f"F1 SCORE: {np.max(result['f1_score'])}")

        print(f"saving...")
        torch.save(
            result,
            'defense_result_roc.pt',
        )
        print(f"complete.")



class BadVPT_STRIP(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")


        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger=torch.load("shallow50_8_badvpt_trigger_modify_eurosat.pt").to(self.device)
        self.model=torch.load("issba_dmlab_model.pt").to(self.device)
        # self.poisonedtestset=torch.load("shallow50_16_wanet_testsets_dmlab.pth")
        self.encoder=torch.load("encoder_dmlab.pt").to(self.device)
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")

    def train_classifier(self, train_loader, val_loader, test_loader):
        logger.info("Test Stage!")
        # self.model.enc.transformer.on = True

        FRR = 0.05
        n_detect = 0
        bckdr_H = []

        sampler = None
        test_loader = torch.utils.data.DataLoader(
            self.testdatasets,
            batch_size=8,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=True,
        )

        # sampler = None
        # poison_loader = torch.utils.data.DataLoader(
        #     self.poisonedtestset,
        #     batch_size=8,
        #     shuffle=False,
        #     sampler=sampler,
        #     num_workers=self.cfg.DATA.NUM_WORKERS,
        #     pin_memory=self.cfg.DATA.PIN_MEMORY,
        #     drop_last=False,
        # )

        backdoor_num=0
        # poison_iter=iter(poison_loader)
        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        for i, input_data in enumerate(test_loader):  # type: ignore
            inputs, labels = self.get_input(input_data)
            inputs=inputs.to(self.device)
            # backdoored_images=next(poison_iter)[0]
            # print(backdoored_images)
            # backdoored_images=self.trigger(inputs).to(self.device)

            X=inputs

            len = X.shape[0]
            pos_len = int(len * 1)
            Y = X[0:pos_len].clone().detach().to(self.device)
            Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
            for j in range(pos_len):
                Y[:, j] = Y[:, j].clone().to(self.device)
                residual = self.encoder([secret, Y[:, j]]).to(self.device)
                Y[:, j] = Y[:, j].clone() + residual

            Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
            backdoored_images=Y.to(self.device)

            defence = STRIP(holdout_x=inputs, model=self.model,device=self.device)
            backdoor_num+=backdoored_images.shape[0]
            for i in range(backdoored_images.shape[0]):
                print(i)
                detected, h = defence.detect_backdoor(backdoored_images[i, ::])
                n_detect += detected
                res=h.cpu().detach().numpy()
                bckdr_H.append(res)

            print(torch.cuda.memory_summary())
            del defence
            del inputs
            del backdoored_images
            torch.cuda.empty_cache()

        print("FRR: ", FRR, " FAR: ", "{0:.2f}".format((1.0 - (n_detect / backdoor_num))))






# def poison(x_train, y_train, param):
#     target_label = param["target_label"]
#     num_images = int(param["poisoning_rate"] * y_train.shape[0])
#
#     if param["clean_label"]:
#         index = np.where(y_train == target_label)
#         index = index[0]
#         index = index[:num_images]
#
#         x_train[index] = poison_frequency(x_train[index], y_train[index], param)
#         return x_train
#
#     else:
#         index = np.where(y_train != target_label)
#         index = index[0]
#         index = index[:num_images]
#         x_train[index] = poison_frequency(x_train[index], y_train[index], param)
#         y_train[index] = target_label
#         return x_train
#
# def poison_frequency(x_train, y_train, param):
#     if x_train.shape[0] == 0:
#         return x_train
#     x_train *= 255.
#     if param["YUV"]:
#         # transfer to YUV channel
#         x_train = RGB2YUV(x_train)
#
#     # transfer to frequency domain
#     x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)
#
#     # plug trigger frequency
#     for i in range(x_train.shape[0]):
#         for ch in param["channel_list"]:
#             for w in range(0, x_train.shape[2], param["window_size"]):
#                 for h in range(0, x_train.shape[3], param["window_size"]):
#                     for pos in param["pos_list"]:
#                         x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]
#
#     x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)
#
#     if param["YUV"]:
#         x_train = YUV2RGB(x_train)
#     x_train /= 255.
#     x_train = np.clip(x_train, 0, 1)
#     return x_train
#
#
# def impose(x_train, y_train, param):
#     x_train = poison_frequency(x_train, y_train, param)
#     return x_train




class BadVPT_Ftrojan(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        # self.trigger = Trigger(self.cfg, dtype=torch.float32).cuda()
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger = make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger = make_scheduler(self.optimizer_trigger, self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")




    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1.25 * len(train_loader)  # 修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        sampler=None
        self.poisontrainloader = torch.utils.data.DataLoader(
            self.poisonedtrain,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.poisontestloader = torch.utils.data.DataLoader(
            self.poisonedtest,
            batch_size=32,
            shuffle=(False if sampler else True),
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )

        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(self.poisontrainloader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                # X, targets = self.get_input(input_data)
                X,targets=input_data[0],input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%50==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
                self.eval_classifier(
                    self.poisontestloader, "test", epoch == total_epoch - 1)


class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label





class BadVPT_ISSBA(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.evaluator_backdoor = evaluator

        dataset_name=cfg.DATA.NAME
        from ..data.datasets.tf_dataset import TFDataset
        self.traindataset = TFDataset(cfg, "train")
        self.testdatasets=TFDataset(cfg, "test")
        self.cfg=cfg
        self.target=0

        secret_size = 20

        train_data_set = []
        train_secret_set = []
        for idx, input_data in enumerate(self.traindataset):
            img=input_data["image"].float()
            lab=input_data["label"]
            train_data_set.append(img.tolist())
            secret = np.random.binomial(1, .5, secret_size).tolist()
            train_secret_set.append(secret)
            print(idx)

        # for idx, input_data in enumerate(self.testdatasets):
        #     img=input_data["image"].float()
        #     lab=input_data["label"]
        #     train_data_set.append(img.tolist())
        #     secret = np.random.binomial(1, .5, secret_size).tolist()
        #     train_secret_set.append(secret)
        #     print(idx)
        #     if idx>=5000:
        #         break

        train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)

        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,

            'benign_training': False,
            'batch_size': 16,
            'num_workers': 8,

            'lr': 0,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],

            'epochs': 1,

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 100,

            'save_dir': 'experiments',
            'experiment_name': 'train_poison_DataFolder_CIFAR10_ISSBA'
        }

        encoder_schedule = {
            'secret_size': secret_size,
            'enc_height': 224,
            'enc_width': 224,
            'enc_in_channel': 3,
            'enc_total_epoch': 20,
            'enc_secret_only_epoch': 2,
            'enc_use_dis': False,
        }

        # Configure the attack scheme
        issba =ISSBA(
            dataset_name="cifar100",
            train_dataset=self.traindataset,
            test_dataset=self.testdatasets,
            train_steg_set=train_steg_set,
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            y_target=self.target,
            poisoned_rate=0.2,  # follow the default configure in the original paper
            encoder_schedule=encoder_schedule,
            encoder=None,
            schedule=schedule,
            seed=42,
            deterministic=True
        )

        issba.train(schedule=schedule)

        # self.poisonedtrain=issba.get_poisoned_dataset()

        self.encoder=torch.load("encoder_clevr.pt").to(self.device)
        # self.cln_train_dl=torch.load("cln_train_dl_cifar100.pt")
        # self.bd_train_dl=torch.load("bd_train_dl_cifar100.pt")


    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        self.model.enc.transformer.on = False
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")


    @torch.no_grad()
    def eval_classifier_bad(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        self.model.enc.transformer.on = False
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        self.encoder.to(self.device)
        self.encoder.eval()

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            X=X.view(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
            for j in range(X.shape[0]):
                X[:,j] = X[:,j].to(self.device)
                residual = self.encoder([secret, X[:,j].to(self.device)]).to(self.device)
                X[:,j] = X[:,j].to(self.device) + residual
                # X[:,j] = X[:,j].clamp(0, 1).to(self.device)

            X=X.view(X.shape[1],X.shape[2],X.shape[3],X.shape[4])
            targets=torch.zeros_like(targets)+self.target
            targets.to(self.device)

            data_time.update(time.time() - end)
            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")



    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        # self.warm_up(train_loader)
        logger.info("Train Stage!")
        self.model.eval()
        self.encoder.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)  # 不懂这里是用来做什么的
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training


        for epoch in range(100):
            # reset averagemeters to measure per-epoch results
            self.model.enc.transformer.on=False
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()
            secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                X=X.to(self.device)
                len=X.shape[0]
                pos_len=int(len*0.5)
                Y=X[0:pos_len].clone().detach().to(self.device)
                Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
                for j in range(pos_len):
                    Y[:,j] = Y[:,j].clone().to(self.device)
                    residual = self.encoder([secret, Y[:,j]]).to(self.device)
                    Y[:,j] = Y[:,j].clone() + residual

                Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
                X=torch.cat((X,Y),dim=0).cuda()
                targets =torch.cat((targets,torch.zeros_like(targets[0:pos_len]) + self.target),dim=0).cuda()
                X.to(self.device)
                targets.to(self.device)

                data_time.update(time.time() - end)

                # self.cls_criterion = build_loss(self.cfg, newloss=True)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                # self.cls_criterion = build_loss(self.cfg, newloss=False)
                # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            # Enable eval mode
            self.model.eval()


            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None and (epoch+1)%100==0:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4e')

                prefix="train_part"
                test_name = prefix + "_" + train_loader.dataset.name

                total_logits = []
                total_targets = []
                total=10000
                log_interval = self.cfg.SOLVER.LOG_EVERY_N
                secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
                for idx, input_data in enumerate(test_loader):
                    if self.cfg.DBG and idx == 20:
                        # if debugging, only need to see the first few iterations
                        break

                    X, targets = self.get_input(input_data)
                    X = X.to(self.device)
                    len = X.shape[0]
                    pos_len = int(len * 1)
                    Y = X[0:pos_len].clone().detach().to(self.device)
                    Y = Y.view(1, Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3])
                    for j in range(pos_len):
                        Y[:, j] = Y[:, j].clone().to(self.device)
                        residual = self.encoder([secret, Y[:, j]]).to(self.device)
                        Y[:, j] = Y[:, j].clone() + residual

                    Y = Y.view(Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
                    # X = torch.cat((X, Y), dim=0).cuda()
                    # targets = torch.cat((targets, torch.zeros_like(targets[0:pos_len]) + self.target), dim=0).cuda()
                    targets=torch.zeros_like(targets[0:pos_len]) + self.target

                    Y.to(self.device)
                    targets.to(self.device)

                    data_time.update(time.time() - end)

                    # self.cls_criterion = build_loss(self.cfg, newloss=True)
                    train_loss, _ = self.forward_one_batch(Y, targets, False)
                    # self.cls_criterion = build_loss(self.cfg, newloss=False)
                    # train_loss, _ = self.forward_one_batch_backdoor(X, targets, True)

                    if train_loss == -1:
                        # continue
                        return None

                    losses.update(train_loss.item(), X.shape[0])

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if (idx + 1) % log_interval == 0:
                        logger.info(
                            "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                                idx + 1,
                                total,
                                losses.val,
                                batch_time.val,
                                data_time.val
                            ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                        )

                    total_targets.extend(list(targets.numpy()))
                    total_logits.append(_)

                logger.info(
                    f"Inference ({prefix}):"
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        data_time.avg, batch_time.avg)
                    + "average loss: {:.4f}".format(losses.avg))
                if self.model.side is not None:
                    logger.info(
                        "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
                # total_testimages x num_classes
                joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
                self.evaluator.classify(
                    joint_logits, total_targets,
                    test_name, self.cfg.DATA.MULTILABEL,
                )
        # torch.save(self.model,"issba_caltech101_model.pt")


class BadVPT_DetectionASR(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # logger.info("\tSetting up the optimizer...")
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.cls_criterion = build_loss(self.cfg)

        self.model=torch.load("shallow50_16_badnets_dmlab.pt")
        # self.encoder=torch.load("encoder_caltech101.pt")
        # self.trigger = torch.load("shallow50_8_badvpt_trigger_modify.pt")
        self.poisonedtestset=torch.load("shallow50_16_badnets_testsets_dmlab.pth")
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger=make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger=make_scheduler(self.optimizer_trigger,self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator



    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, features_clean_mode = self.model.enc(inputs)  # (batchsize, num_cls)
            self.model.enc.use_feature = False
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        return loss, outputs, features_clean_mode


    @torch.no_grad()
    def eval_classifier_backdoor_new(self, data_loader,test_loader, prefix, index, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        poison_iter = iter(data_loader)
        for idx, input_data in enumerate(test_loader):
            backdoored_image=next(poison_iter)[0].cuda()
            if idx>=3200:
                break
            if (idx not in index) and (idx+3200 not in index):
                continue
            end = time.time()
            if idx in index and idx + 3200 in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = torch.zeros_like(targets) + 0
                targets = targets.cuda()
                # Y=self.trigger(X)
                Y=backdoored_image

                # Y = X.clone().detach().to(self.device)
                # Y= Y.clone().to(self.device)
                # residual = self.encoder([secret, Y]).to(self.device)
                # Y = Y.clone() + residual

                targets_2=torch.zeros_like(targets) + 0
                X=torch.cat((X,Y))
                targets=torch.cat((targets,targets_2))
            elif idx in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = torch.zeros_like(targets) + 0
                targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            elif idx+3200 in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = targets.cuda()
                targets = torch.zeros_like(targets) + 0
                # X = self.trigger(X)
                # Y = X.clone().detach().to(self.device)
                # Y= Y.clone().to(self.device)
                # residual = self.encoder([secret, Y]).to(self.device)
                # Y = Y.clone() + residual
                # X=Y
                X=backdoored_image


            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def train_classifier(self, train_loader, val_loader, test_loader):
        self.index=torch.load("teco_pred_result_badnets_dmlab.pth")
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        index_choose=[]

        for i in range(self.index.shape[0]):
            if self.index[i]!=1:
                index_choose.append(i)

        sampler=None
        poison_test_loader = torch.utils.data.DataLoader(
            self.poisonedtestset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        # self.model.enc.transformer.on=True
        epoch=99
        total_epoch=100
        print(len(index_choose))
        self.eval_classifier_backdoor_new(poison_test_loader,test_loader, 'test', index_choose,epoch == total_epoch - 1)



class BadVPT_DetectionASR_2(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # logger.info("\tSetting up the optimizer...")
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.cls_criterion = build_loss(self.cfg)

        self.model=torch.load("issba_dmlab_model.pt")
        self.encoder=torch.load("encoder_dmlab.pt")
        # self.trigger = torch.load("shallow50_4_badvpt_trigger_modify_caltech101.pt")
        # self.poisonedtestset=torch.load("shallow50_16_wanet_testsets_dmlab.pth")
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger=make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger=make_scheduler(self.optimizer_trigger,self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator



    def forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            self.model.enc.use_feature = True
            outputs_no, features_clean_mode = self.model.enc(inputs)  # (batchsize, num_cls)
            self.model.enc.use_feature = False
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        return loss, outputs, features_clean_mode


    @torch.no_grad()
    def eval_classifier_backdoor_new(self, data_loader,test_loader, prefix, index, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        # poison_iter = iter(data_loader)
        for idx, input_data in enumerate(test_loader):
            # backdoored_image=next(poison_iter)[0].cuda()
            if idx>=3000:
                break
            if (idx not in index) and (idx+3000 not in index):
                continue
            end = time.time()
            if idx in index and idx + 3000 in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = torch.zeros_like(targets) + 0
                targets = targets.cuda()
                # Y=self.trigger(X)
                # Y=backdoored_image

                Y = X.clone().detach().to(self.device)
                Y= Y.clone().to(self.device)
                residual = self.encoder([secret, Y]).to(self.device)
                Y = Y.clone() + residual

                targets_2=torch.zeros_like(targets) + 0
                X=torch.cat((X,Y))
                targets=torch.cat((targets,targets_2))
            elif idx in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = torch.zeros_like(targets) + 0
                targets = targets.cuda()
                # X = self.trigger(X)
                # X=backdoored_image

                Y = X.clone().detach().to(self.device)
                Y= Y.clone().to(self.device)
                residual = self.encoder([secret, Y]).to(self.device)
                Y = Y.clone() + residual
                X=Y


            # X, targets = self.deal_data(input_data)
            # measure data loading time
            elif idx+3000 in index:
                X, targets = self.get_input(input_data)
                X = X.cuda()
                targets = targets.cuda()
                targets = torch.zeros_like(targets) + 0
                # X = self.trigger(X)


            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")


    def train_classifier(self, train_loader, val_loader, test_loader):
        self.index=np.load("pred_dmlab_issba.npy")
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        index_choose=[]

        for i in range(self.index.shape[0]):
            if self.index[i]!=1:
                index_choose.append(i)

        # sampler=None
        # poison_test_loader = torch.utils.data.DataLoader(
        #     self.poisonedtestset,
        #     batch_size=1,
        #     shuffle=False,
        #     sampler=sampler,
        #     num_workers=self.cfg.DATA.NUM_WORKERS,
        #     pin_memory=self.cfg.DATA.PIN_MEMORY,
        #     drop_last=False,
        # )

        epoch=99
        total_epoch=100
        print(len(index_choose))
        self.eval_classifier_backdoor_new(test_loader,test_loader, 'test', index_choose,epoch == total_epoch - 1)




class BadVPT_NAD(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # logger.info("\tSetting up the optimizer...")
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.cls_criterion = build_loss(self.cfg)

        self.model=torch.load("issba_dmlab_model.pt")
        self.encoder=torch.load("encoder_dmlab.pt")
        # self.trigger = torch.load("shallow50_8_badvpt_trigger_modify_dmlab.pt")
        # self.poisonedtestset=torch.load("shallow50_16_wanet_testsets_dmlab.pth")
        # self.trigger.set_target_name(self.cfg.BACKDOOR.TARGET)
        # self.optimizer_trigger=make_optimizer([self.trigger], self.cfg.SOLVER)
        # self.scheduler_trigger=make_scheduler(self.optimizer_trigger,self.cfg.SOLVER)
        self.evaluator_backdoor = evaluator

        dataset_name = cfg.DATA.NAME
        if dataset_name.startswith("vtab-"):
            from ..data.datasets.tf_dataset import TFDataset
            dataset = TFDataset(cfg, "test")
        from torch.utils.data import random_split
        self.fttrainset, self.fttestset = random_split(dataset, [1000,21735])
        sampler = None
        # Create a loader
        self.ftuningloader = torch.utils.data.DataLoader(
            self.fttrainset,
            batch_size=16,
            shuffle=True,
            sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )
        self.fttestloader = torch.utils.data.DataLoader(
            self.fttestset,
            batch_size=16,
            shuffle=True,
            sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=False,
        )

        # self.poisontestloader = torch.utils.data.DataLoader(
        #     self.poisonedtestset,
        #     batch_size=16,
        #     shuffle=False,
        #     sampler=sampler,
        #     num_workers=cfg.DATA.NUM_WORKERS,
        #     pin_memory=cfg.DATA.PIN_MEMORY,
        #     drop_last=False,
        # )

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + "NAD_backdoor"
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + "NAD_clean"
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    @torch.no_grad()
    def eval_classifier_bad(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        self.model.enc.transformer.on = False
        self.target=0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        self.encoder.to(self.device)
        self.encoder.eval()

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            X=X.view(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
            for j in range(X.shape[0]):
                X[:,j] = X[:,j].to(self.device)
                residual = self.encoder([secret, X[:,j].to(self.device)]).to(self.device)
                X[:,j] = X[:,j].to(self.device) + residual
                # X[:,j] = X[:,j].clamp(0, 1).to(self.device)

            X=X.view(X.shape[1],X.shape[2],X.shape[3],X.shape[4])
            targets=torch.zeros_like(targets)+self.target
            targets.to(self.device)

            data_time.update(time.time() - end)
            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")



    def train_classifier(self, train_loader, val_loader, test_loader):
        logger.info("Test Stage!")

        # print(self.trigger.trigger)
        self.model.eval()
        # self.trigger.eval()
        # self.trigger.cuda()
        # print("!!!!model:",self.model.enc.transformer.prompt_proj.weight.data)
        self.model.cuda()
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = 1 * len(self.ftuningloader)
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training
        deterministic = True
        defense = NAD(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            power=5.0,
            beta = [500, 500],
            target_layers = ['enc', 'head'],
            seed = 42,
            deterministic = deterministic)
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,
            'batch_size': 16,
            'num_workers': 8,
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'tune_lr': 0.01,
            'tune_epochs': 10,
            'epochs': 20,
            'schedule': [4, 8, 12, 16],
            'log_iteration_interval': 20,
            'test_epoch_interval': 20,
            'save_epoch_interval': 20,
            'save_dir': 'experiments/NAD-defense',
            'experiment_name': 'NAD_VIT'
            }
        dataset_name = self.cfg.DATA.NAME
        if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
            from ..data.datasets.tf_dataset import TFDataset
        dataset = TFDataset(self.cfg, "test")
        self.model.enc.transformer.on = False
        defense.repair(dataset=dataset, portion=0.25, schedule=schedule)
        self.model = defense.get_model()
        if self.fttestloader is not None:
            epoch = 100
        # self.model.enc.transformer.on = False
        # self.evaluator.update_iteration(epoch)
        # self.evaluator_backdoor.update_iteration(epoch)
        self.eval_classifier(
        self.fttestloader, "test", epoch == 100)
        # self.eval_classifier_backdoor_onclean(self.fttestloader, 'test', epoch == 100)
        self.eval_classifier_bad(test_loader, 'test', epoch == 100)
        # self.model.enc.transformer.on = True
        # self.eval_classifier(
        # self.fttestloader, "test", epoch == 100)
        # self.eval_classifier_backdoor_onclean(self.fttestloader, 'test', epoch == 100)
        # self.eval_classifier_backdoor(self.fttestloader, 'test', epoch == 100)


class BadVPT_IBAU(Trainer):
    def __init__(
            self,
            cfg: CfgNode,
            model: nn.Module,
            evaluator: Evaluator,
            device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = torch.load("shallow50_16_wanet_dmlab.pt")
        # self.trigger = torch.load("shallow50_4_badvpt_trigger_modify_caltech101.pt")
        # self.encoder=torch.load("encoder_dmlab.pt")
        self.poisonedtestset=torch.load("shallow50_16_wanet_testsets_dmlab.pth")
        self.device = device

        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.optimizer.lr = 0.001
        # print(self.optimizer)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.cls_criterion = build_loss(self.cfg)
        # self.optimizer=torch.optim.SGD(self.model.parameters(),lr=1,momentum=0.9,weight_decay=0.001)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        self.evaluator_backdoor = evaluator

        self.cls_criterion = build_loss(self.cfg)

        # # Creating data indices for training and validation splits:
        dataset_name = cfg.DATA.NAME
        if dataset_name.startswith("vtab-"):
            # import the tensorflow here only if needed
            from ..data.datasets.tf_dataset import TFDataset
            self.dataset = TFDataset(cfg, "test")

    def new_forward_one_batch_backdoor(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, outputs

    @torch.no_grad()
    def eval_classifier_backdoor(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        # test_name = prefix + "_" + data_loader.dataset.name
        test_name = prefix + "_" + "fine_tuning_test"
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tBackdoor Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Backdoor Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    @torch.no_grad()
    def eval_classifier_backdoor_onclean(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        # test_name = prefix + "_" + data_loader.dataset.name
        test_name = prefix + "_" + "fine_tuning_test"
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.cuda()
            targets = targets.cuda()
            # X, targets = self.deal_data(input_data)
            # measure data loading time
            # targets = torch.zeros_like(targets) + self.trigger.target
            X = self.trigger(X)
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTrigger Test {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Trigger Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator_backdoor.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    def get_results(self, model, criterion, data_loader, device='cuda'):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, targets = self.get_input(batch)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            return correct / total

    @torch.no_grad()
    def eval_classifier_bad(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        self.model.enc.transformer.on = False
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        self.target=0

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        secret = torch.FloatTensor(np.random.binomial(1, .5, 20).tolist()).to(self.device)
        self.encoder.to(self.device)
        self.encoder.eval()

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if isinstance(input_data,dict):
                X, targets = self.get_input(input_data)
            else:
                X,targets=input_data[0],input_data[1]
            # measure data loading time
            X=X.view(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
            for j in range(X.shape[0]):
                X[:,j] = X[:,j].to(self.device)
                residual = self.encoder([secret, X[:,j].to(self.device)]).to(self.device)
                X[:,j] = X[:,j].to(self.device) + residual
                # X[:,j] = X[:,j].clamp(0, 1).to(self.device)

            X=X.view(X.shape[1],X.shape[2],X.shape[3],X.shape[4])
            targets=torch.zeros_like(targets)+self.target
            targets.to(self.device)

            data_time.update(time.time() - end)
            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")



    def train_classifier(self, train_loader, val_loader, test_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Test Stage!")
        # print(self.trigger.trigger)
        self.model.eval()
        # self.trigger.eval()
        # self.trigger.cuda()
        # print("!!!!model:",self.model.enc.transformer.prompt_proj.weight.data)
        self.model.cuda()
        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        # total_data =1*len(self.ftuningloader)    #修改为两倍长度
        best_epoch = -1
        best_metric = 0
        best_metric_backdoor = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")

        # start repair
        logger.info('==> Preparing data..')
        # test_set, att_val_set, unl_set = get_eval_data(self.dataset, attack_name='badnets', target_lab='8', args=args)

        unl_set, test_set = random_split(self.dataset, [1000, 21735])
        # data loader for verifying the clean test accuracy
        clnloader = torch.utils.data.DataLoader(
            test_set, batch_size=8, shuffle=False, num_workers=1)

        # data loader for verifying the attack success rate
        poiloader = torch.utils.data.DataLoader(
            self.poisonedtestset, batch_size=8, shuffle=False, num_workers=1)

        # data loader for the unlearning step
        unlloader = torch.utils.data.DataLoader(
            unl_set, batch_size=8, shuffle=False, num_workers=1)

        optim = 'Adam'
        lr = 0.0005
        n_rounds = 5
        K = 5

        # self.model.enc.transformer.on = False
        criterion = nn.CrossEntropyLoss()
        if optim == 'SGD':
            outer_opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optim == 'Adam':
            outer_opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        ACC = self.get_results(self.model, criterion, clnloader, device)
        # ASR = self.get_results(self.model, criterion, poiloader, device)

        print('Original ACC:', ACC)

        # print('Original ASR:', ASR)

        def loss_inner(perturb, model_params):
            images = images_list[0].to(device)
            labels = labels_list[0].long().to(device)
            #     per_img = torch.clamp(images+perturb[0],min=0,max=1)
            per_img = images + perturb[0]
            per_logits = self.model.forward(per_img)
            loss = F.cross_entropy(per_logits, labels, reduction='none')
            loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
            return loss_regu

        ### define the outer loss L1
        def loss_outer(perturb, model_params):
            portion = 0.01
            images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
            patching = torch.zeros_like(images, device='cuda')
            number = images.shape[0]
            rand_idx = random.sample(list(np.arange(number)), int(number * portion))
            patching[rand_idx] = perturb[0]
            #     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
            unlearn_imgs = images + patching
            logits = self.model(unlearn_imgs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            return loss

        images_list, labels_list = [], []
        for index, batch in enumerate(unlloader):
            images, labels = self.get_input(batch)
            images_list.append(images)
            labels_list.append(labels)
        inner_opt = hg.GradientDescent(loss_inner, 0.1)

        ### inner loop and optimization by batch computing
        logger.info("=> Conducting Defence..")
        self.model.eval()
        # ASR_list = [self.get_results(self.model, criterion, poiloader, device)]
        ACC_list = [self.get_results(self.model, criterion, clnloader, device)]
        shape = (1, 3, 224, 224)
        temp_shape = torch.zeros(shape)

        for round in range(n_rounds):
            batch_pert = torch.zeros_like(temp_shape, requires_grad=True, device='cuda')
            batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

            for index, batch in enumerate(unlloader):
                images, labels = self.get_input(batch)
                images = images.to(device)
                ori_lab = torch.argmax(self.model.forward(images), axis=1).long()
                #         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
                per_logits = self.model.forward(images + batch_pert)
                loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
                loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
                batch_opt.zero_grad()
                loss_regu.backward(retain_graph=True)
                batch_opt.step()

            # l2-ball
            # pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
            pert = batch_pert

            # unlearn step
            # self.model.requires_grad_(True)
            self.model.enc.transformer.requires_grad_(True)
            # canshu=list(self.model.parameters())
            self.model.enc.transformer.encoder.requires_grad_(False)
            self.model.enc.transformer.prompt_embeddings_badone.requires_grad_(False)
            canshu = list(self.model.enc.transformer.parameters())
            canshu2=[]
            for i in range(len(canshu)):
                canshu2.append(canshu[i])
                if i>=0:
                    break
            for batchnum in range(len(images_list)):
                outer_opt.zero_grad()
                hg.fixed_point(pert, canshu2, K, inner_opt, loss_outer)
                outer_opt.step()

            # ASR_list.append(self.get_results(self.model,criterion,poiloader,device))
            ACC_list.append(self.get_results(self.model, criterion, clnloader, device))
            print('Round:', round)

            print('ACC:', self.get_results(self.model, criterion, clnloader, device))
            # print('ASR:',self.get_results(self.model,criterion,poiloader,device))

        if clnloader is not None:
            epoch = 100
            # self.model.enc.transformer.on = False
            # self.evaluator.update_iteration(epoch)
            # self.evaluator_backdoor.update_iteration(epoch)
            self.eval_classifier(
                clnloader, "test", epoch == 100)
            self.eval_classifier(
                poiloader, "test", epoch == 100)
            # self.eval_classifier_backdoor_onclean(clnloader, 'test', epoch == 100)
            # self.eval_classifier_backdoor(clnloader, 'test', epoch == 100)
            # self.model.enc.transformer.on = True
            # self.eval_classifier(
            #     clnloader, "test", epoch == 100)
            # self.eval_classifier_backdoor_onclean(clnloader, 'test', epoch == 100)
            # self.eval_classifier_backdoor(clnloader, 'test', epoch == 100)