for step in 150; do
    python verl-main/scripts/model_merger.py --backend fsdp --hf_model_path /cache/data/huggingface_models/Qwen2.5-VL-7B-Instruct --local_dir /cache/exps/verl_checkpoints/captioner/captioner_qwen7b_reasoner_200_nokl_no_filter_virl39k_llm_ds_7b_no_cap_icl_0.9/global_step_$step/actor --target_dir /cache/exps/verl_checkpoints/verl_grpo_caption_qwen25_vl_7b_reasoner_200_nokl_no_filter_high_0.3_llm_ds_7b_virl39k_no_cap_icl_0.9_high_0.25_step${step} 
done

for step in 200; do
    python verl-main/scripts/model_merger.py --backend fsdp --hf_model_path /cache/data/huggingface_models/Qwen2.5-VL-7B-Instruct --local_dir /cache/exps/verl_checkpoints/reasoner/reasoner_qwen7b_virl39k_no_kl_no_filter_high_0.3/global_step_$step/actor --target_dir /cache/exps/verl_grpo_reason_qwen25_vl_7b_no_kl_no_filter_high_0.3_step${step} 
done
