# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
from tqdm import tqdm
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
import pandas as pd
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from torch.utils.data import DataLoader


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer, hf_processor
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'
    

    dataset = RLHFDataset(parquet_files=config.data.path,
                                            tokenizer=tokenizer,
                                            processor=processor,
                                            prompt_key=config.data.prompt_key,
                                            image_key=config.data.get('image_key', 'images'),
                                            max_prompt_length=config.data.max_prompt_length,
                                            return_raw_chat=config.data.get('return_raw_chat', False),
                                            truncation=config.data.get('truncation', 'error'),
                                            filter_overlong_prompts=config.data.filter_overlong_prompts,
                                            num_workers=config.data.get('filter_overlong_prompts_workers', None))

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    # total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size

    sampler = SequentialSampler(data_source=dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=collate_fn,
        sampler=sampler
    )

    output_lst = [[] for _ in range(config.data.n_samples)]
    # num_batch = -(-total_samples // config_batch_size)
    num_batch = len(dataloader)
    # for batch_idx in range(num_batch):
    for batch_idx, batch_dict in enumerate(dataloader):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        # batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]

        # batch_dict = collate_fn(items)

        data = DataProto.from_single_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES

        for i in range(config.data.n_samples):
            if 'multi_modal_inputs' in data.non_tensor_batch.keys():
                gen_batch = data.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            output = wg.generate_sequences(gen_batch)

            # remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=True)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst[i].extend(output_text_unpad)

    import ipdb
    ipdb.set_trace()
    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    from pandas import read_parquet
    data = read_parquet("/home/ma-user/work/cache/data/datasets/virl39k_no_deepscaler_caption.parquet")
    data['responses'] = output_lst
    
    import datasets
    data_hf = datasets.Dataset.from_pandas(data)
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    data_hf.to_parquet(config.data.output_path)


    # # add to the data frame
    # dataset[f'responses'] = output_lst

    # # write to a new parquet
    # output_dir = os.path.dirname(config.data.output_path)
    # makedirs(output_dir, exist_ok=True)
    # dataset.to_parquet(config.data.output_path)

    return output_text


if __name__ == '__main__':
    main()
