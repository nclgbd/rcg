#!/bin/bash
# bash scripts/run_main_rdm.sh

config_name=rdm
config_dir=$(pwd)/config
timestamp="$(date +%Y-%m-%d_%H:%M:%S)"
log_filename="${timestamp}_${config_name}"
log_dir="outputs/$config_name/logs"
mkdir -p $log_dir

nohup python main_rdm.py --config-dir=$(pwd)/config/ --config-name=$config_name \
    device=1 \
    >"$log_dir/$log_filename.log" &
