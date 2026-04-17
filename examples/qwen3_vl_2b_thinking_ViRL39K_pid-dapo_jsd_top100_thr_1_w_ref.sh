#!/bin/bash

set -x

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

EXP_NAME=qwen3_vl_2b_thinking_ViRL39K_pid-dapo_jsd_top100_thr_1_w_ref
SAVE_PATH=/mnt/vlm-ks3/xiangshizhe/vlm-rl-xsz/${EXP_NAME}
PROJECT_NAME=VLM-RL-Xiaomi-xsz
export CUDA_LAUNCH_BLOCKING=1
export SWANLAB_MODE="offline"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.9
export SWANLAB_DIR=${SAVE_PATH}/swanlog  # Correct env variable name for PAPO_qwen3
export RAY_DEDUP_LOGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${SWANLAB_DIR}  # Ensure directory exists

MODEL_PATH=/mnt/vlm-ks3/fengfeng/model_zoo/Qwen3-VL-2B-Thinking
TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=2048
MAX_HINT_LENGTH=4096

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
    data.max_hint_prompt_length=${MAX_HINT_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.rollout.n=5 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.val_freq=40 \
    trainer.val_before_train=False \
    algorithm.enable_pid=true \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_coef=1.0e-2 \
    algorithm.kl_direction=forward_kl \
    algorithm.pid_threshold=1.0 \
    algorithm.pid_top_k=100 \
    algorithm.pid_coef=5.0e-2 \
    algorithm.pid_kl_direction=jsd_kl \
    algorithm.online_filtering=false \




