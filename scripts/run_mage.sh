#!/bin/bash
# . scripts/run_mage.sh

timestamp="$(date +%Y-%m-%d_%H:%M:%S)"
config_name="rcg"
log_filename="${timestamp}_${config_name}"
log_dir="outputs/$config_name/logs"
mkdir -p $log_dir

nohup python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    main_mage.py \
    --pretrained_enc_arch mocov3_vit_base \
    --pretrained_enc_path pretrained_enc_ckpts/mocov3/vitb.pth.tar --rep_drop_prob 0.1 \
    --use_rep --rep_dim 256 --pretrained_enc_withproj --pretrained_enc_proj_dim 256 \
    --rdm_steps 250 --eta 1.0 --temp 6.0 --num_iter 20 --num_images 50000 --cfg 0.0 \
    --batch_size 64 --input_size 256 \
    --model mage_vit_base_patch16 \
    --mask_ratio_min 0.5 --mask_ratio_max 1.0 --mask_ratio_mu 0.75 --mask_ratio_std 0.25 \
    --epochs 200 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${IMAGENET_DIR} \
    --dist_url tcp://${MASTER_SERVER_ADDRESS}:2214
