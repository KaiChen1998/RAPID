from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_local_rank_and_world_size, get_gpu_memory, auto_split_flag
import json
from vllm import LLM, SamplingParams
from .visual_utils import process_vision_info
from .utils import reorganize_prompt, load_image
from transformers import AutoTokenizer, AutoModel
from ...smp import *


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map







class Qwen2VLOmniCap(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        tensor_parallel_size = 2,
        function_type = 'qa',
        best_of_n=1,
        # reward_model_path=None,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
       
        self.system_prompt = system_prompt
        self.verbose = verbose
        from transformers import AutoProcessor
        assert model_path is not None
        self.model_path = model_path
        self.function_type = function_type
        
        local_rank, world_size = get_local_rank_and_world_size()
        if world_size > 1:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,  
                gpu_memory_utilization=0.8,      
                dtype="bfloat16",
                # device=f'cuda:{local_rank}',
                mm_processor_kwargs={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels} if 'OmniCaptioner' in model_path else None,
                limit_mm_per_prompt={"image": 38},
                distributed_executor_backend="external_launcher",
                seed=0,
                trust_remote_code=True,
            )
        else:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.8,      
                dtype="bfloat16",
                mm_processor_kwargs={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels} if 'OmniCaptioner' in model_path else None,
                limit_mm_per_prompt={"image": 38},
                trust_remote_code=True,
            )  


        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
        )


        self.best_of_n = best_of_n

        if self.best_of_n > 1:
            self.sampling_params_tts = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_new_tokens,
                n=best_of_n-1
            )
        self.max_new_tokens = max_new_tokens


    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None, function_type: str | None = None, qa=None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """

        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
              
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                if  function_type =='OCR_Chart' or function_type =='OCR_chart_caption': 
                    
                    if function_type =='OCR_Chart':
                        item = {'type': 'text', 'text': 'Convert this chart into a table in Markdown format.'}
                    else:
                        item = {'type': 'text', 'text': 'Describe this chart in detail.'}
                elif function_type == 'OCR_table':
                    p = 0.5
                    import random
                    if random.random() < p:
                        item = {'type': 'text', 'text': 'Produce a table in LaTeX format based on this table.'}
                    else:
                        item = {'type': 'text', 'text': "Please analyze this table in detail."}
                elif function_type == 'query_cond_3':
                    item = {'type': 'text', 'text': f'Question: ' + s['value'] + "\n" + "Please describe the image. DO NOT try to answer the question!"}
                elif function_type == 'qa':
                    item = {'type': 'text', 'text': s['value']}
                elif function_type == 'qa_cot':
                    item = {'type': 'text', 'text': s['value'].replace('Please try to answer the question with short words or phrases if possible.','')}
                elif function_type == 'qa_cot_step':
                    item = {'type': 'text', 'text': s['value'].replace('Please try to answer the question with short words or phrases if possible.','') + " Please think step by step. The final answer MUST BE put in \\boxed{}."}
                elif function_type == 'revisual':
                    item = {'type': 'text', 'text': s['value'] + "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."} 
                elif function_type == 'holistic':
                    item = {'type': 'text', 'text': 'Describe this image in detail.'}
                else:
                    raise NotImplementedError

            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
    
    def _prepare_content_internvl(self, inputs: list[dict[str, str]], dataset: str | None = None, function_type: str | None = None, qa=None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """

        img_cnt = sum([1 if item['type'] == 'image' else 0 for item in inputs])
        texts = []
        images = []
        for s in inputs:
            if s['type'] == 'image':
                # item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                item = ensure_image_url(s['value'])
                images.append(item)
            elif s['type'] == 'text':
                if function_type == 'qa_cot_step':
                    # item = {'type': 'text', 'value': s['value'].replace('Please try to answer the question with short words or phrases if possible.','') + " Please think step by step. The final answer MUST BE put in \\boxed{}."}
                    item = s['value'].replace('Please try to answer the question with short words or phrases if possible.','') + " Please think step by step. The final answer MUST BE put in \\boxed{}."
                elif function_type == "qa_cot":
                    item = s['value'].replace('Please try to answer the question with short words or phrases if possible.','')
                elif function_type == "query_cond_3":
                    item = 'Question: ' + s['value'] + "\n" + "Please describe the image. DO NOT try to answer the question!"
                else:
                    assert False
                item = "".join([f"<image>\n" for _ in range(img_cnt)]) + item
                texts.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            

        
        return images, texts
    

       
    def _prepare_input(self, message, dataset=None,question_type=None,answer=None, question=None, caption2=None, caption=None, qa=None):
        messages = []

        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})

        if 'InternVL' in self.model_path or 'internvl' in self.model_path:
            images, texts = self._prepare_content_internvl(message, dataset=dataset,function_type=self.function_type, qa=qa)
            for text in texts:
                messages.append({'role': 'user', 'content': text})
        else:
            messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset,function_type=self.function_type, qa=qa)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if 'InternVL' in self.model_path or 'internvl' in self.model_path:
            from PIL import Image
            images = [Image.open(image[7:]) for image in images]
        else:
            images, _ = process_vision_info([messages])
        
        if images:
            return {
                'multi_modal_data': {'image': images},
                'prompt': text[0],
            }
        else:
            return {
                'prompt': text[0],
            }

