#!/usr/bin/env python

import datetime
import hydra
import json
import numpy as np
import os
import time
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_rdm import train_one_epoch
from rdm.util import instantiate_from_config
from config import RCGConfiguration

# rtk
from rtk._datasets import cxr14
from rtk.utils import get_logger, hydra_instantiate, _strip_target

logger = get_logger("main_rdm")

# def get_args_parser():
#     parser = argparse.ArgumentParser("RDM training", add_help=False)
#     parser.add_argument(
#         "--batch_size",
#         default=64,
#         type=int,
#         help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
#     )
#     parser.add_argument("--epochs", default=400, type=int)
#     parser.add_argument(
#         "--accum_iter",
#         default=1,
#         type=int,
#         help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
#     )

#     # config
#     parser.add_argument("--input_size", default=256, type=int, help="images input size")

#     parser.add_argument("--config", type=str, help="config file")

#     # Optimizer parameters
#     parser.add_argument(
#         "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
#     )

#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=None,
#         metavar="LR",
#         help="learning rate (absolute lr)",
#     )
#     parser.add_argument(
#         "--blr",
#         type=float,
#         default=1e-3,
#         metavar="LR",
#         help="base learning rate: absolute_lr = base_lr * total_batch_size",
#     )
#     parser.add_argument(
#         "--min_lr",
#         type=float,
#         default=0.0,
#         metavar="LR",
#         help="lower lr bound for cyclic schedulers that hit 0",
#     )
#     parser.add_argument(
#         "--cosine_lr", action="store_true", help="Use cosine lr scheduling."
#     )
#     parser.add_argument("--warmup_epochs", default=0, type=int)

#     # Dataset parameters
#     parser.add_argument(
#         "--data_path",
#         default="/home/nicoleg/workspaces/dissertation/.data/CHEST_XRAY_14/",
#         type=str,
#         help="dataset path",
#     )

#     parser.add_argument(
#         "--output_dir",
#         default="./outputs",
#         help="path where to save, empty for no saving",
#     )
#     parser.add_argument(
#         "--log_dir", default="./outputs/logs", help="path where to tensorboard log"
#     )
#     parser.add_argument(
#         "--device", default=0, help="device to use for training / testing"
#     )
#     parser.add_argument("--seed", default=0, type=int)
#     parser.add_argument("--resume", default="", help="resume from checkpoint")

#     parser.add_argument(
#         "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
#     )
#     parser.add_argument("--num_workers", default=10, type=int)
#     parser.add_argument(
#         "--pin_mem",
#         action="store_true",
#         help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
#     )
#     parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
#     parser.set_defaults(pin_mem=True)

#     # distributed training parameters
#     parser.add_argument(
#         "--world_size", default=1, type=int, help="number of distributed processes"
#     )
#     parser.add_argument("--local_rank", default=-1, type=int)
#     parser.add_argument("--dist_on_itp", action="store_true")
#     parser.add_argument(
#         "--dist_url", default="env://", help="url used to set up distributed training"
#     )

#     return parser


@hydra.main(config_path="config", config_name="rdm")
def main(args: RCGConfiguration):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    misc.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.getcwd()))
    logger.info("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # simple augmentation
    # transform_train = transforms.Compose(
    #     [
    #         transforms.Resize(256, interpolation=3),
    #         transforms.RandomCrop(256),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ]
    # )

    # dataset_train = datasets.ImageFolder(
    #     os.path.join(args.data_path, "train"), transform=transform_train
    # )
    datasets = cxr14.load_cxr14_dataset(cfg=args)
    dataset_train = datasets[0]

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        logger.info("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # data_loader_train = dataloaders[0]

    # load model config
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    # set arguments generation params
    args.class_cond = config.model.params.get("class_cond", False)

    model.to(device)

    model_without_ddp = model
    logger.info("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size

    logger.info("base lr: %.2e" % (args.lr / eff_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Log parameters
    params = list(model_without_ddp.model.parameters())
    params = params + list(model_without_ddp.cond_stage_model.parameters())
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: {}M".format(n_params / 1e6))
    if global_rank == 0:
        log_writer.add_scalar("num_params", n_params / 1e6, 0)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        misc.save_model_last(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    # args = get_args_parser()
    # args = args.parse_args()
    main()
