#!/bin/bash

set -x

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

EXP_NAME=qwen3_vl_8b_thinking_ViRL39K_ref_pid-grpo_jsd_top100_5e-2_thr_1_rollout_8_max_token_4096
SAVE_PATH=your/save/path/${EXP_NAME}
PROJECT_NAME=""

export SWANLAB_MODE="offline"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.9
export SWANLAB_DIR=${SAVE_PATH}/swanlog
export RAY_DEDUP_LOGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${SWANLAB_DIR}

MODEL_PATH=your/model/path/Qwen3-VL-8B-Thinking
TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

CONGI_FILE="examples/configs/config_grpo.yaml"
TRAIN_FILE="/mnt/llm-plus-public/dataset/PAPO_ViRL39K_train_with_hint/data"
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
    data.prompt_with_hint_key=prompt_with_hint \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.rollout.n=8 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.val_freq=40 \
    algorithm.enable_pid=true \
    algorithm.pid_threshold=1.0 \
    algorithm.pid_top_k=100 \
    algorithm.pid_coef=5.0e-2 \
    algorithm.pid_kl_direction=jsd_kl \
    algorithm.pid_use_ref_teacher=true
