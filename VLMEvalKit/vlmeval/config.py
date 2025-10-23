from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '



qwen2vl_series_diy = {

    'MLLM': partial(Qwen2VLOmniCap, \
          model_path='path_to_ckpt',  \
          tensor_parallel_size=1, \
          max_new_tokens=8192, \
          temperature=0, \
          system_prompt = 'You are a helpful assistant.',
          function_type = 'qa_cot'
      ),

    'LLM': partial(Reasoner, \
          model_path='path_to_ckpt',  \
          tensor_parallel_size=1, \
          max_new_tokens=8192, \
          temperature=0.6, \
          top_p=0.95, \
          function_type="joint" \
       ),

#     'Captioner-Qwen-2-5-7B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-7B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),

#     'Qwen-2-5-VL-7B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-7B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot',
#           best_of_n=8,
#           ),

#     'Qwen-2-5-VL-3B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-3B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot',
#           best_of_n=8,
#           ),

#     'Captioner-Qwen-2-5-VL-3B-reason-virk39k-250': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_reason_qwen25_vl_3b_llm_ds_7b_virl39k_step250',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3',
#           ),

#     'Qwen-2-5-VL-3B-reason-virk39k-250': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_reason_qwen25_vl_3b_virl39k_step250',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot'
#           ),

#     'Captioner-Qwen-2-5-VL-3B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-3B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3',
#           ),


#     'Qwen-2-5-VL-32B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-32B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot',
#           best_of_n=4,
#           ),

#     'Qwen-2-5-VL-32B-reason-virk39k-100': partial(Qwen2VLOmniCap, \
#           model_path='/cache/exps/verl_checkpoints/verl_grpo_reason_qwen25_vl_32b_virl39k_step100',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot'
#           ),


#     'Captioner-Qwen-2-5-VL-32B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-VL-32B-Instruct',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3',
#           ),

#     'VL-Rethinker-7B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/VL-Rethinker-7B',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot'
#           ),

#     'Qwen-2.5-VL-7B-Captioner-with-qa-sefl-virl39k-llm-ds-7b-no-sys': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_with_qa_rethinker_7b_llm_ds_7b_virl39k_step60',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'query_cond_3'
#           ),


#     'Captioner-Qwen-2-5-7B-virl39k-llm-ds-7b-140': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_qwen25_vl_7b_virl39k_llm_ds_7b_step140',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=2048, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),


#     'Captioner-Qwen-2-5-3B-virl39k-llm-ds-7b-200': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_qwen25_vl_3b_llm_ds_7b_virl39k_step200',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),

#     'VL-Rethinker-Captioner-with-qa-virl39k-100-llm-ds-7b-no-sys': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_with_qa_rethinker_7b_virl39k_llm_ds_7b_no_sys_step100',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'query_cond_3'
#           ),

#     'Captioner-VL-Rethinker-7B': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/VL-Rethinker-7B',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),

#     'Qwen-2-5-7B-mmk12-300': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/checkpoints/verl_grpo_caption_qwen_25_vl_7b_mmk12_step300',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot'
#           ),


#     'Captioner-Qwen-2-5-7B-mmk12-300': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/checkpoints/verl_grpo_caption_qwen_25_vl_7b_mmk12_step300',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0.7, \
#           top_p=0.85, \
#           top_k=20, \
#           repetition_penalty=1.2, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),


#     'VL-Rethinker-Captioner-Reasoner': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/VL-Rethinker-7B',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'cap_qa_cot'
#           ),

#     'VL-Rethinker-Captioner-with-qa-virl39k-160-llm-ds-7b': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_with_qa_rethinker_7b_llm_ds_7b_virl39k_step160',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'query_cond_4'
#           ),

#     'VL-Rethinker-Captioner-virl39k-120-llm-ds-7b': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_rethinker_7b_llm_ds_7b_virl39k_step120',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.',
#           function_type = 'query_cond_3'
#           ),


#     'VL-Rethinker-virl39k-120-llm-ds-7b': partial(Qwen2VLOmniCap, \
#           model_path='/home/ma-user/work/cache/exps/verl_checkpoints/verl_grpo_caption_rethinker_7b_llm_ds_7b_virl39k_step120',  \
#           min_pixels=2*28*28, max_pixels=6400*28*28, \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0, \
#           system_prompt = 'You are a helpful assistant.',
#           function_type = 'qa_cot'
#           ),
#     'VisualPRM': partial(VisualPRM, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/VisualPRM-8B-v1.1',  \
#           function_type = 'qa_cot', \
#           ),

#     'VisualPRM-Caption': partial(VisualPRM, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/VisualPRM-8B-v1.1',  \
#           function_type = 'query_cond_3', \
#           ),

#     'Reasoner-deepseek-distill-1.5B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B',  \
#           tensor_parallel_size=1, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95 \
#           ),

#     'Reasoner-deepseek-distill-7B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/DeepSeek-R1-Distill-Qwen-7B',  \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Reasoner-Qwen3-8B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen3-8B',  \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Reasoner-Qwen3-4B': partial(Reasoner, \
#           model_path='/cache/Qwen3-4B',  \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),


#     'Reasoner-deepseek-distill-32B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/DeepSeek-R1-Distill-Qwen-32B',  \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Reasoner-QwQ-32B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/QwQ-32B',  \
#           tensor_parallel_size=4, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Reasoner-Qwen3-235B-A22B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen3-235B-A22B',  \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Reasoner-Qwen3-32B': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen3-32B',  \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           function_type="joint" \
#           ),

#     'Qwen2.5-32B-Instruct': partial(Reasoner, \
#           model_path='/home/ma-user/work/cache/data/huggingface_models/Qwen2.5-32B-Instruct',  \
#           tensor_parallel_size=8, \
#           max_new_tokens=8192, \
#           temperature=0.6, \
#           top_p=0.95, \
#           ),


}


supported_VLM = {}

model_groups = [
    qwen2vl_series_diy
]

for grp in model_groups:
    supported_VLM.update(grp)
