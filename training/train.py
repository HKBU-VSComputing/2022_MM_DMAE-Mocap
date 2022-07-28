import os
import time
import json
import torch
import torch.backends.cudnn as cudnn
import datetime
import argparse
import random
import numpy as np
from pathlib import Path
from timm.models import create_model

from training.core.utils import NativeScalerWithGradNormCount as NativeScaler
from training.core.utils import TensorboardLogger
from training.core.utils import init_distributed_mode as util_init_distributed_mode
from training.core.utils import get_rank as util_get_rank
from training.core.utils import seed_worker as util_seed_worker
from training.core.utils import get_world_size as util_get_world_size
from training.core.utils import cosine_scheduler as util_cosine_scheduler
from training.core.utils import save_model as util_save_model
from training.core.utils import is_main_process as util_is_main_process
from training.core.utils import auto_load_model as util_auto_load_model

from training.core.optim_factory import create_optimizer
from training.core.engine_for_openpose import train_one_epoch_v3
from training.dataset.dataset_skel3d import ShelfSkelDataset

from model.dmae import dmae_v5


def get_args():
    parser = argparse.ArgumentParser('OpenPose limb decoder training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='openpose_mae_ver3', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    # learning rate
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # lr warm up protocol
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Output config
    parser.add_argument('--output_dir', default='./openpose',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./openpose/logger',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False
    )

    return model


def main(args):
    util_init_distributed_mode(args)
    print(args)

    # cude config
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + util_get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.mask_ratio = 0.5
    args.rot = False
    args.random_rot = True
    args.output_dir = "./output/shelf_base/"
    args.log_dir = "./output/shelf_base/"
    args.fine_tune = False
    dataset_train = ShelfSkelDataset(
        ["./data/shelf_3d_train_normalized.npy",
         "./data/shelf_3d_test_normalized.npy", ],
        train=True, mask_ratio_t=args.mask_ratio, mask_ratio_j=args.mask_ratio,
        rot=args.rot, random_rot=args.random_rot)
    dataset_test = ShelfSkelDataset(
        ["./data/shelf_3d_train_normalized.npy",
         "./data/shelf_3d_test_normalized.npy", ],
        train=False, mask_ratio_t=args.mask_ratio, mask_ratio_j=args.mask_ratio,
        rot=args.rot)
    print('ShelfSkelDataset has {} items'.format(len(dataset_train)))
    skel_data, (mask_idx, unmask_idx) = dataset_train[0]
    patch_num, patch_dim = skel_data.shape
    joint_num = dataset_train.joint_num
    frame_num = dataset_train.window_size
    mask_ratio = dataset_train.mask_ratio_j

    # model
    model = dmae_v5(patch_num, patch_dim, frame_num, mask_ratio=mask_ratio)

    # get logger
    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    # get data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=util_seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s - %s" % (str(model), str(model_without_ddp)))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * util_get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = util_cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = util_cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    util_auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                         loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch_v3(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            normlize_target=args.normlize_target,
            finetune=args.fine_tune,
            test_loader=dataset_test,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                util_save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and util_is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
