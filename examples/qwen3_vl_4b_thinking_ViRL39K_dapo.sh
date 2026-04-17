#!/bin/bash

set -x

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

EXP_NAME=qwen3_vl_4b_thinking_ViRL39K_dapo
SAVE_PATH=/mnt/vlm-ks3/xiangshizhe/vlm-rl-xsz/${EXP_NAME}
PROJECT_NAME=VLM-RL-Xiaomi-xsz

export SWANLAB_MODE="offline"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.9
export SWANLAB_DIR=${SAVE_PATH}/swanlog  # Correct env variable name for PAPO_qwen3
export RAY_DEDUP_LOGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


mkdir -p ${SWANLAB_DIR}  # Ensure directory exists

MODEL_PATH=/mnt/vlm-ks3/fengfeng/model_zoo/Qwen3-VL-4B-Thinking
TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=2048


CONGI_FILE="examples/configs/config_grpo.yaml"
TRAIN_FILE="/mnt/llm-plus-public/dataset/PAPO_ViRL39K_train/data"
VAL_FILE="/mnt/llm-plus-public/dataset/PAPO_MMK12_test/data"

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/qwen3_vl_think.py:compute_score"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.rollout.n=5 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.val_freq=40 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.enable_pid=false
    
    

