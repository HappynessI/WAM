#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [ -n "${ROBOTWIN_ROOT:-}" ]; then
    :
elif [ -f "$SCRIPT_DIR/../../script/eval_policy.py" ]; then
    ROBOTWIN_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
elif [ -f "$SCRIPT_DIR/../../RoboTwin/script/eval_policy.py" ]; then
    ROBOTWIN_ROOT=$(cd -- "$SCRIPT_DIR/../../RoboTwin" && pwd)
elif [ -f /data0/code/RoboTwin/script/eval_policy.py ]; then
    ROBOTWIN_ROOT=/data0/code/RoboTwin
else
    echo "Cannot find RoboTwin root. Set ROBOTWIN_ROOT=/path/to/RoboTwin." >&2
    exit 1
fi

TASK_NAME=${1:-adjust_bottle}
TASK_CONFIG=${2:-demo_clean_skele}
MODEL_NAME=${3:-robotwin_contact}
CHECKPOINT_ID=${4:-10000}

SEED=${SEED:-42}
GPU_ID=${GPU_ID:-0}
TEST_NUM=${TEST_NUM:-1}
POLICY_NAME=${POLICY_NAME:-frappe}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-False}
RDT_STEP=${RDT_STEP:-30}

if [ -x /home/sjy/anaconda3/envs/RoboTwin/bin/python ]; then
    DEFAULT_PYTHON=/home/sjy/anaconda3/envs/RoboTwin/bin/python
elif [ -x /home/sjy/anaconda3/envs/wam/bin/python ]; then
    DEFAULT_PYTHON=/home/sjy/anaconda3/envs/wam/bin/python
else
    DEFAULT_PYTHON=python
fi
CONDA_PYTHON=${CONDA_PYTHON:-$DEFAULT_PYTHON}

if [ -d "$SCRIPT_DIR/checkpoints" ]; then
    DEFAULT_CHECKPOINT_ROOT="$SCRIPT_DIR/checkpoints"
else
    DEFAULT_CHECKPOINT_ROOT="$ROBOTWIN_ROOT/policy/frappe/checkpoints"
fi
export FRAPPE_CHECKPOINT_ROOT=${FRAPPE_CHECKPOINT_ROOT:-$DEFAULT_CHECKPOINT_ROOT}
DEFAULT_RDT_WEIGHTS_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)/weights/RDT
if [ -d "$DEFAULT_RDT_WEIGHTS_ROOT" ]; then
    export FRAPPE_RDT_WEIGHTS_ROOT=${FRAPPE_RDT_WEIGHTS_ROOT:-$DEFAULT_RDT_WEIGHTS_ROOT}
fi

ALIAS_ROOT=${FRAPPE_ALIAS_ROOT:-/tmp/frappe_robotwin_policy}
mkdir -p "$ALIAS_ROOT"
ln -sfn "$SCRIPT_DIR" "$ALIAS_ROOT/$POLICY_NAME"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export USE_TF=0
export USE_FLAX=0
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
FRAPPE_USE_WAM_DEPS=${FRAPPE_USE_WAM_DEPS:-0}
if [ "$FRAPPE_USE_WAM_DEPS" = "1" ]; then
    FRAPPE_PYDEPS_ROOT=${FRAPPE_PYDEPS_ROOT:-/tmp/frappe_pydeps}
    WAM_SITE_PACKAGES=${WAM_SITE_PACKAGES:-/home/sjy/anaconda3/envs/wam/lib/python3.10/site-packages}
    mkdir -p "$FRAPPE_PYDEPS_ROOT"
    if [ -d "$WAM_SITE_PACKAGES" ]; then
        for dep in \
            peft peft-*.dist-info \
            accelerate accelerate-*.dist-info \
            transformers transformers-*.dist-info \
            tokenizers tokenizers-*.dist-info \
            huggingface_hub huggingface_hub-*.dist-info \
            safetensors safetensors-*.dist-info \
            diffusers diffusers-*.dist-info; do
            for src in $WAM_SITE_PACKAGES/$dep; do
                [ -e "$src" ] || continue
                ln -sfn "$src" "$FRAPPE_PYDEPS_ROOT/$(basename "$src")"
            done
        done
        export PYTHONPATH="$ALIAS_ROOT:$FRAPPE_PYDEPS_ROOT:${PYTHONPATH:-}"
    else
        export PYTHONPATH="$ALIAS_ROOT:${PYTHONPATH:-}"
    fi
else
    export PYTHONPATH="$ALIAS_ROOT:${PYTHONPATH:-}"
fi

echo -e "\033[33mRoboTwin root: ${ROBOTWIN_ROOT}\033[0m"
echo -e "\033[33mPolicy dir: ${SCRIPT_DIR}\033[0m"
echo -e "\033[33mCheckpoint root: ${FRAPPE_CHECKPOINT_ROOT}\033[0m"
echo -e "\033[33mGPU id (to use): ${GPU_ID}\033[0m"
echo -e "\033[33mTest episodes: ${TEST_NUM}\033[0m"

cd "$ROBOTWIN_ROOT"

PYTHONWARNINGS=ignore::UserWarning "$CONDA_PYTHON" script/eval_policy.py \
    --config "$SCRIPT_DIR/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting "$MODEL_NAME" \
    --seed "$SEED" \
    --checkpoint_id "$CHECKPOINT_ID" \
    --policy_name "$POLICY_NAME" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --test_num "$TEST_NUM" \
    --eval_video_log "$EVAL_VIDEO_LOG" \
    --rdt_step "$RDT_STEP"
