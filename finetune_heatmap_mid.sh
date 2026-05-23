#!/bin/bash

TASK_NAME="adjust_bottle"
CONFIG_NAME="heatmap_mid_train_$TASK_NAME"
CONFIG_FILE="model_config/$CONFIG_NAME.yml"

echo "CONFIG_FILE_PATH: $CONFIG_FILE"

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export WANDB_PROJECT="WAM_HEATMAP_DEPTH"
export WANDB_DEFAULT_RUN_NAME=$CONFIG_NAME

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file $CONFIG_FILE does not exist!"
  exit 1
fi

PRETRAINED_MODEL_NAME=$(python scripts/read_yaml.py "$CONFIG_FILE" pretrained_model_name_or_path)
TRAIN_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" train_batch_size)
SAMPLE_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_batch_size)
MAX_TRAIN_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" max_train_steps)
CHECKPOINTING_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpointing_period)
SAMPLE_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_period)
CHECKPOINTS_TOTAL_LIMIT=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoints_total_limit)
LEARNING_RATE=$(python scripts/read_yaml.py "$CONFIG_FILE" learning_rate)
DATALOADER_NUM_WORKERS=$(python scripts/read_yaml.py "$CONFIG_FILE" dataloader_num_workers)
STATE_NOISE_SNR=$(python scripts/read_yaml.py "$CONFIG_FILE" state_noise_snr)
GRAD_ACCUM_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" gradient_accumulation_steps)
OUTPUT_DIR=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoint_path)
CUDA_USE=$(python scripts/read_yaml.py "$CONFIG_FILE" cuda_visible_device)
WM_INPUT_TYPE=$(python scripts/read_yaml.py "$CONFIG_FILE" wm_input_type)
WM_TARGET_TYPE=$(python scripts/read_yaml.py "$CONFIG_FILE" wm_target_type)
TCP_HISTORY_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" tcp_history_size)
FUTURE_BINS=$(python scripts/read_yaml.py "$CONFIG_FILE" future_bins)
FUTURE_BIN_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" future_bin_size)
HEATMAP_LOSS_WEIGHT=$(python scripts/read_yaml.py "$CONFIG_FILE" heatmap_loss_weight)
DEPTH_LOSS_WEIGHT=$(python scripts/read_yaml.py "$CONFIG_FILE" depth_loss_weight)
PRECOMPUTE_LANG=$(python scripts/read_yaml.py "$CONFIG_FILE" precompute_lang_embeddings)
LANG_EMBED_DESC_TYPES=$(python scripts/read_yaml.py "$CONFIG_FILE" lang_embed_desc_types)
LANG_EMBED_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" lang_embed_batch_size)

PRETRAINED_MODEL_NAME=$(echo "$PRETRAINED_MODEL_NAME" | tr -d '"')
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')
WM_INPUT_TYPE=$(echo "$WM_INPUT_TYPE" | tr -d '"')
WM_TARGET_TYPE=$(echo "$WM_TARGET_TYPE" | tr -d '"')

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  echo "Created output directory: $OUTPUT_DIR"
else
  echo "Output directory already exists: $OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_USE

if [ "$PRECOMPUTE_LANG" = "True" ] || [ "$PRECOMPUTE_LANG" = "true" ]; then
  python scripts/precompute_robotwin_lang_embeds.py \
    --model_config_path=$CONFIG_FILE \
    --text_encoder=$TEXT_ENCODER_NAME \
    --desc_types=$LANG_EMBED_DESC_TYPES \
    --batch_size=$LANG_EMBED_BATCH_SIZE
fi

python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME --wm_horizon 30

accelerate launch --main_process_port=28499 --multi_gpu --num_processes=2 main_mid_train.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_period=$CHECKPOINTING_PERIOD \
    --sample_period=$SAMPLE_PERIOD \
    --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
    --lr_scheduler="constant" \
    --learning_rate=$LEARNING_RATE \
    --mixed_precision="bf16" \
    --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    --dataset_type="finetune" \
    --state_noise_snr=$STATE_NOISE_SNR \
    --load_from_hdf5 \
    --report_to=wandb \
    --precomp_lang_embed \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --model_config_path=$CONFIG_FILE \
    --CONFIG_NAME=$CONFIG_NAME \
    --enc_type="theia-base-vit" \
    --resolution=256 \
    --proj_coeff=0.05 \
    --encoder-depth=21 \
    --learnable_tokens=196 \
    --wm_horizon=30 \
    --wm_input_type=$WM_INPUT_TYPE \
    --wm_target_type=$WM_TARGET_TYPE \
    --tcp_history_size=$TCP_HISTORY_SIZE \
    --future_bins=$FUTURE_BINS \
    --future_bin_size=$FUTURE_BIN_SIZE \
    --heatmap_loss_weight=$HEATMAP_LOSS_WEIGHT \
    --depth_loss_weight=$DEPTH_LOSS_WEIGHT
