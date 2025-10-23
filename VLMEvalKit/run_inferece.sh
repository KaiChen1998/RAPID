export vlmeval_outputs_dir=/cache/vlmeval_outputs

cd ./VLMEvalKit
HF_MODEL_ROOT=/cache/data/huggingface_models
CKPT_ROOT=/cache/exps/verl_checkpoints/
LLM_FUNC_TYPE=joint
CAP_FUNC_TYPE=query_cond_3
QA_FUNC_TYPE=qa_cot_step
for MLLM in verl_grpo_caption_qwen25_vl_7b_reasoner_200_nokl_no_filter_high_0.3_llm_ds_7b_virl39k_no_cap_icl_0.9_high_0.25_step150; do
 for REASON_MODEL in Qwen3-8B ; do
      DATA_LIST="MathVista_MINI LogicVista WeMath MathVerse_MINI_Vision_Only DynaMath MMMU_DEV_VAL MathVision"
      ########## QA
      torchrun run.py --data ${DATA_LIST} --model MLLM --model_path ${CKPT_ROOT}/${MLLM} --tensor_parallel_size 1 --function_type ${QA_FUNC_TYPE} --max_new_tokens 4096  --suffix _${QA_FUNC_TYPE} --work-dir ${vlmeval_outputs_dir}/${MLLM} --api_nproc 32 --mode infer
      ######### Caption
      torchrun run.py --data ${DATA_LIST} --model MLLM --model_path ${CKPT_ROOT}/${MLLM} --tensor_parallel_size 1 --function_type ${CAP_FUNC_TYPE} --max_new_tokens 4096 --suffix _${CAP_FUNC_TYPE} --system_prompt "You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question." --work-dir  ${vlmeval_outputs_dir}/${MLLM} --mode infer

      QA_LIST=""
      CAP_LIST=""
      for data in ${DATA_LIST}; do
          QA_FILE=${vlmeval_outputs_dir}/${MLLM}/MLLM/MLLM_${data}_${QA_FUNC_TYPE}.csv
          [ -f "$QA_FILE" ] && echo "OK" || echo "$QA_FILE does not exist"
          CAP_FILE=${vlmeval_outputs_dir}/${MLLM}/MLLM/MLLM_${data}_${CAP_FUNC_TYPE}.csv
          [ -f "$CAP_FILE" ] && echo "OK" || echo "$CAP_FILE does not exist"
          QA_LIST+="$QA_FILE "
          CAP_LIST+="$CAP_FILE "
      done

      # # ########## Joint 
      torchrun --nproc_per_node 8 run.py --data ${DATA_LIST} --model LLM --model_path ${HF_MODEL_ROOT}/${REASON_MODEL} --tensor_parallel_size 1 --function_type ${LLM_FUNC_TYPE}  --suffix _${REASON_MODEL}_${LLM_FUNC_TYPE} --qa-file $QA_LIST --caption-file $CAP_LIST  --work-dir  ${vlmeval_outputs_dir}/${MLLM}  --api_nproc 32 --mode infer --reuse

  done
done