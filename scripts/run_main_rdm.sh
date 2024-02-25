#!/bin/bash
# bash scripts/run_main_rdm.sh

config_name=rdm
config_dir=$(pwd)/config
timestamp="$(date +%Y-%m-%d_%H:%M:%S)"
log_filename="${timestamp}_${config_name}"
log_dir="outputs/$config_name/logs"
mkdir -p $log_dir

nohup python main_rdm.py --config-dir=$(pwd)/config/ --config-name=$config_name -m \
    device=0 blr=0.00001,0.000001 weight_decay=0.0,0.0005,0.005 datasets/encoding=class-conditioned-encoding datasets/transforms=rdm-transforms datasets/preprocessing=diffusion-preprocessing \
    datasets.target=class_conditioned_labels datasets.dim=256 datasets.preprocessing.positive_class=null datasets.dataloader.batch_size=512 >"$log_dir/$log_filename.log" &
