#!/bin/bash

export NCCL_ALGO=Tree
# export DS_SKIP_CUDA_CHECK=1
# export WANDB_MODE= "offline"
# export WANDB_API_KEY="8390dc80ee401927129fe80b7cdc6e1d46d0638e"

# pip install -U transformers accelerate
# pip install --upgrade Pillow
# pip install git+https://github.com/Dao-AILab/causal-conv1d

# need change 4 place
experiment_name=test_models--FreedomIntelligence--LongLLaVAMed_jamba-9B
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log
# export CUDA_VISIBLE_DEVICES='4,5,6,7'
# CUDA_VISIBLE_DEVICES="4,5,6,7"
## For MultiNode
# ssh check
# apt-get install pdsh

# chown root:root /usr/lib/x86_64-linux-gnu/pdsh
# chown root:root /usr/lib
# chmod 755 /usr/lib/x86_64-linux-gnu/pdsh
# chmod 755 /usr/lib

deepspeed --include=localhost:0,1,2,3,4,5,6,7 \
    --hostfile hostfile \
    /home/ly/LLMs/Cervical-RG/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path /home/ly/LLMs/hf_hubs/models--FreedomIntelligence--LongLLaVAMed_jamba-9B \
    --version jamba \
    --vision_tower vit3d \
    --pretrain_mm_mlp_adapter /home/ly/LLMs/Cervical-RG/ckpt/projector/mm_projector.bin \
    --data_path /home/zwding/ly/LongLLaVA3D/json_data/train_data/image_3D_all_del-stage.json \
    --pretrain_vision_model /home/ly/Nas/ly/LLM_ckpt/ckpts/Cervice_CLIP/modified_model.bin \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./ckpts/${experiment_name} \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 15600 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb > ${log_folder}/${log_name} 2>&1 &
    # --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \

    # --lora_enable True --lora_r 256 --lora_alpha 512 --mm_projector_lr 2e-5 \
        # --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    # --mm_projector_type mlp2x_gelu \
    # --resamplePooling 2d \
    #  --tune_mm_mlp_adapter True \
    #--save_steps 10000 \
    # --pretrain_mm_mlp_adapter /home/zwding/ly/LongLLaVA3D/ckpts/all_finetune_15epoch_3D_without_stage3171_Our_CLIP_projector/mm_projector.bin \
    # --tune_mm_mlp_adapter True \
