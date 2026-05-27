#!/bin/bash
# Multi-node launcher for verl/Ray on CloudML-style platforms.
#
# Platform-injected env vars (same command on every node, but values differ):
#   RANK          - this node's rank (0 == master)
#   WORLD_SIZE    - total number of nodes
#   MASTER_ADDR   - master node IP
#   MASTER_PORT   - free port on master (reused for ray head GCS)
#   RESOURCE_GPU  - GPUs per node (e.g. 8)
#
# Usage (entry command identical on all nodes):
#   bash examples/configs/launch_multinode.sh examples/configs/<train_script>.sh
#
# Behaviour:
#   - rank 0  : `ray start --head`, wait until WORLD_SIZE*RESOURCE_GPU GPUs
#               have registered, then `bash <train_script>.sh` (which calls
#               `python -m verl.trainer.main ...`).
#   - rank >0 : `ray start --address=...` and block until job ends.

set -uo pipefail
set -x

TRAIN_SCRIPT="${1:?usage: launch_multinode.sh <train_script.sh>}"

: "${RANK:?RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${RESOURCE_GPU:?RESOURCE_GPU not set}"

RAY_HEAD_PORT="${MASTER_PORT}"
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_HEAD_PORT}"

export RAY_ADDRESS="${RAY_HEAD_ADDR}"
# NCCL NVLS init can fail on some multi-node Hopper/NVSwitch clusters.
# Keep it overridable, but default to disabling NVLS so the job can fall back
# to standard NCCL transports instead of looping on transport/nvls.cc errors.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ray stop -f >/dev/null 2>&1 || true

if [[ "${RANK}" == "0" ]]; then
    echo "[launch] rank=0 -> starting Ray HEAD on ${RAY_HEAD_ADDR}"
    ray start --head \
        --node-ip-address="${MASTER_ADDR}" \
        --port="${RAY_HEAD_PORT}" \
        --num-gpus="${RESOURCE_GPU}" \
        --dashboard-host=0.0.0.0

    EXPECTED_GPUS=$(( RESOURCE_GPU * WORLD_SIZE ))
    echo "[launch] waiting for ${WORLD_SIZE} nodes / ${EXPECTED_GPUS} GPUs to register ..."
    for i in $(seq 1 120); do
        CUR_GPUS=$(ray status 2>/dev/null \
                   | grep -Eo '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU' \
                   | head -n1 | awk -F'/' '{print $2}' | awk '{print int($1)}')
        CUR_GPUS=${CUR_GPUS:-0}
        echo "[launch] ray sees ${CUR_GPUS}/${EXPECTED_GPUS} GPUs"
        if [[ "${CUR_GPUS}" -ge "${EXPECTED_GPUS}" ]]; then
            break
        fi
        sleep 5
    done

    echo "[launch] cluster ready, starting training: ${TRAIN_SCRIPT}"
    bash "${TRAIN_SCRIPT}"
    TRAIN_EXIT=$?

    ray stop -f || true
    exit ${TRAIN_EXIT}
else
    echo "[launch] rank=${RANK} -> joining Ray cluster at ${RAY_HEAD_ADDR}"
    for i in $(seq 1 120); do
        if ray health-check --address="${RAY_HEAD_ADDR}" >/dev/null 2>&1; then
            break
        fi
        echo "[launch] head not ready yet, retry ${i}/120 ..."
        sleep 5
    done

    ray start --address="${RAY_HEAD_ADDR}" \
        --num-gpus="${RESOURCE_GPU}" \
        --block
fi
