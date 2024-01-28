#!/bin/bash
# . scripts/train_rcg.sh

timestamp="$(date +%Y-%m-%d_%H:%M:%S)"
config_name="chest-xray14-diffusion"
log_filename="${timestamp}_${config_name}"
log_dir="outputs/$config_name/logs"
mkdir -p $log_dir

nohup python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    main_rdm.py \
    --config config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml \
    --batch_size 128 --input_size 256 \
    --epochs 200 \
    --blr 1e-6 --weight_decay 0.01 \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${IMAGENET_DIR} \
    --dist_url tcp://${MASTER_SERVER_ADDRESS}:2214
>"$log_dir/$log_filename.log" &
