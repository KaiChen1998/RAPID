from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag
import json
from vllm import LLM, SamplingParams
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



class VisualPRM(BaseModel):

    def __init__(
        self,
        model_path: str,
        verbose: bool = False,
        function_type = 'qa',
    ):
        super().__init__()
      
        self.verbose = verbose
        assert model_path is not None
        self.model_path = model_path
        self.function_type = function_type
        
        local_rank, world_size = get_rank_and_world_size()
        self.device = f'cuda:{local_rank}'
        import sys
        sys.path.append("/home/ma-user/work/cache/data/huggingface_models/VisualPRM-8B-v1.1")

        from modeling_internvl_chat import InternVLRewardModel  # Adjust class name if different
        self.reward_model = InternVLRewardModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).eval().to(self.device)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False)

    
    def _prepare_input_prm(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        self.max_num=8
        self.total_max_num = 64
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))

        content = []
        for s in message:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            if s['type'] == 'text':
                if self.function_type == 'query_cond_3':
                    item = {'type': 'text', 'value': f'Question: ' + s['value'] + "\n" + "Please describe the image. DO NOT try to answer the question!"}
                elif self.function_type == 'query_cond_4':
                    item = {'type': 'text', 'value': f'Question: ' + s['value'] + "\n" + "Please describe the image."}
                elif self.function_type == 'qa':
                    item = {'type': 'text', 'value': s['value']}
                elif self.function_type == 'qa_cot':
                    item = {'type': 'text', 'value': s['value'].replace('Please try to answer the question with short words or phrases if possible.','')}
                else:
                    assert False
            content.append(item)

        prompt = reorganize_prompt(content, image_num, dataset=dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        return prompt, pixel_values, num_patches_list
