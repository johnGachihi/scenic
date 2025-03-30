# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import util.lr_sched as lr_sched
import util.misc as misc
from timm.data import Mixup
from timm.utils import accuracy
import torchmetrics

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric = torchmetrics.MetricCollection({
        "mIoU": torchmetrics.classification.JaccardIndex(
            task="multiclass", num_classes=2, average="macro", ignore_index=-1),
        "IoU_per_class": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=2, average=None, ignore_index=-1),
    }).to(device)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, ret_dataloader in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if args.dataset == "sen1floods11":
            samples, labels = ret_dataloader
            res = torch.ones((samples.shape[0],)).to(device)
        else:
            (samples, res, _, target_res, labels), metadata = ret_dataloader

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda", enabled=args.use_amp):
            outputs = model(samples, input_res=res)
            if args.dataset == "sen1floods11":
                outputs = outputs.permute(0, 2, 3, 1)  # channel last
                output_tmp = outputs.contiguous().view(-1, outputs.size(3))

                target_tmp = targets.permute(0, 2, 3, 1)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1)  # cross entropy loss expects a 1D tensor
                target_tmp = target_tmp.long()
            else:
                output_tmp = outputs
                target_tmp = targets

            loss = criterion(output_tmp, target_tmp)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if args.use_amp:
            loss /= accum_iter
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_iter_step + 1) % accum_iter == 0,
            )
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:  # full precision
            loss /= accum_iter
            # loss.requires_grad = True
            loss.backward()
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()

        if args.dataset == "sen1floods11":
            outputs = outputs.permute(0, 3, 1, 2)  # channel first
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs = outputs.argmax(dim=1)
            targets = targets.squeeze(1)

        score = metric(outputs, targets)

        metric_logger.update(loss=loss_value)

        for key in score.keys():
            if score[key].dim() > 0:  # 1-D array
                for i, s in enumerate(score[key]):
                    metric_logger.meters[f"{key}-{i}"].update(s.item())
            else:
                metric_logger.meters[key].update(score[key].item())

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    eval_base_resolution=1.0,
    gsd_embed=False,
    eval_scale=512,
    reference_size=512,
    args=None
):
    gsd_ratio = eval_base_resolution
    if gsd_embed:
        gsd_ratio = gsd_ratio * (reference_size / eval_scale)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    metric = torchmetrics.MetricCollection({
        "mIoU": torchmetrics.classification.JaccardIndex(
            task="multiclass", num_classes=2, average="macro", ignore_index=-1),
        "IoU_per_class": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=2, average=None, ignore_index=-1),
    }).to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for (samples, labels) in metric_logger.log_every(data_loader, 10, header):
        images = samples
        target = labels
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(
                images,
                input_res=torch.ones(len(images)).float().to(images.device) * gsd_ratio,
            )
            if args.dataset == "sen1floods11":
                output = output.permute(0, 2, 3, 1)
                output_tmp = output.contiguous().view(-1, output.size(3))
                target_tmp = target.permute(0, 2, 3, 1)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1)
                target_tmp = target_tmp.long()
            else:
                output_tmp = output
                target_tmp = target

            loss = criterion(output_tmp, target_tmp)


        if args.dataset == "sen1floods11":
            output = output.permute(0, 3, 1, 2)
            output = torch.nn.functional.softmax(output, dim=1)
            target = target.squeeze(1)

        score = metric(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        for key in score.keys():
            if score[key].dim() > 0:  # 1-D array
                for i, s in enumerate(score[key]):
                    metric_logger.meters[f"{key}-{i}"].update(s.item())
            else:
                metric_logger.meters[key].update(score[key].item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    test_metric = metric.compute()
    logging_text = " "
    for key in test_metric.keys():
        if score[key].dim() > 0:  # 1-D array
            for i, s in enumerate(score[key]):
                logging_text += f"{key}-{i} {s.item():.3f} "
        else:
            logging_text += f"{key} {test_metric[key].item():.3f} "

    logging_text += f"loss {metric_logger.loss.global_avg:.3f}"

    print(logging_text)

    # we can compute global average on all except the metric values
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # we replace the metrics with metric.compute() values
    for key in test_metric.keys():
        if score[key].dim() > 0:  # 1-D array
            for i, s in enumerate(score[key]):
                return_dict[f"{key}-{i}"] = s.item()
        else:
            return_dict[key] = test_metric[key].item()

    return return_dict

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
