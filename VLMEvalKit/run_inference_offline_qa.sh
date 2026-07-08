#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VLMEVAL_OUTPUTS_DIR="${VLMEVAL_OUTPUTS_DIR:-/data/kchenbf/verl_runs/vlmeval_outputs}"
export LMUData="${LMUData:-/data/kchenbf/data/LMUData}"
MODEL_PATH="${MODEL_PATH:-/data/kchenbf/models/Qwen3-VL-2B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DATA_LIST="${DATA_LIST:-MMMU_DEV_VAL WeMath}"
QA_FUNC_TYPE="${QA_FUNC_TYPE:-qa_cot_step}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
API_NPROC="${API_NPROC:-4}"

WORK_DIR="${VLMEVAL_OUTPUTS_DIR}/${MODEL_NAME}"
mkdir -p "${WORK_DIR}"
mkdir -p "${LMUData}"

python run.py \
  --data ${DATA_LIST} \
  --model MLLM \
  --model_path "${MODEL_PATH}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --function_type "${QA_FUNC_TYPE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --suffix "_${QA_FUNC_TYPE}_offline" \
  --work-dir "${WORK_DIR}" \
  --judge exact_matching \
  --api_nproc "${API_NPROC}" \
  --mode all
