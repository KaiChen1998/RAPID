#!/usr/bin/env bash
set -euo pipefail

########################
# # Install dependency
# cd ../VLMEvalKit
# pip install -e .
########################

export LMUData="${LMUData:-/data/bucket-pangu-green/chenkai/data/LMUData}"
DATA_LIST="${DATA_LIST:-MMMU_DEV_VAL}"
MODEL_PATH="${MODEL_PATH:-/data/kchenbf/models/Qwen3-VL-2B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
QA_FUNC_TYPE="${QA_FUNC_TYPE:-qa_cot_step}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
API_NPROC="${API_NPROC:-4}"

WORK_DIR="${MODEL_PATH}/vlmeval_outputs"
mkdir -p "${WORK_DIR}"

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
