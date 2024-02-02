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

# from torch.utils.tensorboard import SummaryWriter

import mlflow

# :huggingface: timm
import timm

# azureml
from azureml.core import Experiment
from azureml.tensorboard import Tensorboard

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_rdm import train_one_epoch
from rdm.util import instantiate_from_config
from config import RCGConfiguration

# rtk
from rtk._datasets import cxr14
from rtk.config import DatasetConfiguration
from rtk.mlflow import prepare_mlflow
from rtk.utils import get_logger, hydra_instantiate, _strip_target

logger = get_logger("main_rdm")


def get_params(cfg: RCGConfiguration, **kwargs):
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", cfg.datasets)
    preprocessing_cfg = dataset_cfg.preprocessing
    params = dict()

    # NOTE: dataset parameters
    def __collect_dataset_params():
        params["dataset_name"] = dataset_cfg.name
        params.update(preprocessing_cfg)
        if params["use_sampling"] == False:
            del params["sampling_method"]

    __collect_dataset_params()

    # in case '_target_' somehow wasn't found
    try:
        del params["_target_"]
    except KeyError:
        pass

    tags = cfg.job.get("tags", {})
    logger.info("Logged parameters:\n{}".format(OmegaConf.to_yaml(params)))
    mlflow.log_params(params)

    if any(tags):
        mlflow.set_tags(tags)

    return params


@hydra.main(config_path="config", config_name="rdm", version_base=None)
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

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    datasets = cxr14.load_cxr14_dataset(cfg=args)
    dataset_train = datasets[0]

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        logger.info("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    data_loader_train = hydra_instantiate(
        cfg=args.datasets.dataloader,
        dataset=dataset_train,
        pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        shuffle=True,
    )

    # load model config
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    # set arguments generation params
    args.class_cond = config.model.params.get("class_cond", args.class_cond)

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
    logger.info("Number of trainable parameters: {} M".format(n_params / 1e6))
    if global_rank == 0:
        mlflow.log_param("num_params", n_params / 1e6)
        # log_writer.add_scalar("num_params", n_params / 1e6, 0)

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

    if mlflow.active_run() is not None:
        mlflow.end_run()

    start_run_kwargs: dict = prepare_mlflow(args)
    start_time = time.time()

    with mlflow.start_run(**start_run_kwargs):
        params = get_params(args)
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
                # log_writer=log_writer,
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
            mlflow.log_metrics(log_stats)

            if args.output_dir and misc.is_main_process():
                # if log_writer is not None:
                #     log_writer.flush()
                with open(
                    os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        mlflow.log_artifacts(args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    mlflow.end_run()


if __name__ == "__main__":
    # args = get_args_parser()
    # args = args.parse_args()
    main()
