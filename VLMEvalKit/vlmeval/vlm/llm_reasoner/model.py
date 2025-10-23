from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from ..omnicaptioner.prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_local_rank_and_world_size, get_gpu_memory, auto_split_flag
import json
from vllm import LLM, SamplingParams

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)


joint_check_prompt = """In the following text, you will receive a detailed caption of an image and a relevant question. In addition, you will be provided with a tentative model response. You goal is to answer the question using these information.    
                        
### The detailed caption of the provided image: {image_caption_batch}

### Note that the caption might contain incorrect solutions, do not be misguided by them.

### A problem to be solved: {problem_prompt}

### A tentaive model response.  {qa_batch}
 
### Note that the above tentative resposne might be inaccurate (due to calculation errors, incorrect logic/reasoning and so on), under such a case, please ignore it and give your own solutions. However, if you do not have enough evidence to show it is wrong, please output the tentative resposne."""


class Reasoner(Qwen2VLPromptMixin, BaseModel):

    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        top_p=0.001,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        tensor_parallel_size = 2,
        function_type = 'joint',
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)

        self.system_prompt = system_prompt
        self.verbose = verbose
        assert model_path is not None
        self.model_path = model_path

        from transformers import AutoTokenizer
        local_rank, world_size = get_local_rank_and_world_size()
        if world_size > 1:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,  
                gpu_memory_utilization=0.9,      
                dtype="bfloat16",
                # device=f'cuda:{local_rank}',
                distributed_executor_backend="external_launcher",
                seed=0
            )
        else:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,  
                gpu_memory_utilization=0.9,      
                dtype="bfloat16",
                max_num_seqs=512,
                # enforce_eager=True,

            )  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if 'gpt' in self.model_path:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
            self.system_gpt = SystemContent(reasoning_effort=ReasoningEffort('High'))
        else:
            stop_token_ids = []
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        self.function_type = function_type

    def _prepare_input_gpt(self, message, caption=None, qa=None):
        for msg in message:
            if msg['type'] == 'text':
                break
        problem = msg['value'].replace('Please try to answer the question with short words or phrases if possible.','').replace('Please try to answer the question with short words or phrases if possible.','')
        prompt = joint_check_prompt.format(image_caption_batch=caption, problem_prompt=problem, qa_batch=qa)
        prompt = prompt + "\n\n### Please think step by step. The final answer MUST BE put in \\boxed{}."

        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, self.system_gpt),
                Message.from_role_and_content(Role.USER, prompt),
            ]
        )
        
        prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        return {'prompt_token_ids': prefill_ids}
 
    def _prepare_input(self, message, dataset=None,question_type=None,answer=None, question=None, caption2=None,caption=None, qa=None):
        messages = []
        
        messages.append({'role': 'system', 'content': "You are a helpful assistant."})
        messages.append({'role': 'user', 'content': self._prepare_summarizer_content
        (message,caption2,caption,qa,dataset=None, function_type=self.function_type)})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


        return {'prompt': text}

    def _prepare_summarizer_content(self, inputs: list[dict[str, str]], image_caption_batch2: list, image_caption_batch: list, qa_batch: list, dataset: str | None = None,  function_type: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        
        has_image = False
        content = []
        for s in inputs:
            if s['type'] == 'image':
                has_image = True
        for s in inputs:
            if s['type'] == 'text':
                if function_type == 'joint':
                
                    if not has_image:
                        item = s['value'].replace('Please try to answer the question with short words or phrases if possible.','').replace('Please try to answer the question with short words or phrases if possible.','')  + ' Please think step by step. The final answer MUST BE put in \\boxed{}.'
                    else:
                        # joint
                        prompt = joint_check_prompt


                        problem = s['value'].replace('Please try to answer the question with short words or phrases if possible.','').replace('Please try to answer the question with short words or phrases if possible.','')
                        item = prompt.format(image_caption_batch=image_caption_batch, problem_prompt=problem, qa_batch=qa_batch)
                        item = item + "\n\n### Please think step by step. The final answer MUST BE put in \\boxed{}."

                else:
                    raise NotImplementedError
                content.append(item)
        return content[0]