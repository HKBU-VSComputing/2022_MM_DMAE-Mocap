import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F

from .utils import MetricLogger, SmoothedValue


def train_one_epoch_v3(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                       normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                       lr_schedule_values=None, wd_schedule_values=None,
                       finetune=False,
                       test_loader: Iterable = None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('test_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # process
        ####################################
        skel_data, (mask_idx, unmask_idx) = batch_data[:]
        skel_data = skel_data.to(device).to(torch.float32)
        mask_idx = mask_idx.to(device)
        unmask_idx = unmask_idx.to(device)

        with torch.cuda.amp.autocast():
            outputs, labels, mask_idx, loss_ = model(skel_data, (mask_idx, unmask_idx))
            # euclidean distance + mse
            loss = torch.mean(F.pairwise_distance(outputs, labels, p=2)) + loss_

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # test
        with torch.no_grad():
            test_skel_data, (test_mask_idx, test_unmask_idx) = test_loader[step % test_loader.data_length]
            test_skel_data = torch.from_numpy(test_skel_data).to(device).unsqueeze(0).to(torch.float32)
            test_mask_idx = torch.from_numpy(test_mask_idx).to(device).unsqueeze(0)
            test_unmask_idx = torch.from_numpy(test_unmask_idx).to(device).unsqueeze(0)
            test_outputs, test_labels, _, test_loss_mse = model(test_skel_data, (test_mask_idx, test_unmask_idx))
            # euclidean distance + mse
            test_loss = torch.mean(F.pairwise_distance(test_outputs, test_labels, p=2)) + test_loss_mse
            test_loss = test_loss.item()
            metric_logger.update(test_loss=test_loss)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
