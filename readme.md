
# RAPID: Reasoning-Aligned Perception Decoupling for Scalable Multi-modal Reasoning

This repository provides training, inference, and evaluation instructions for RAPID.

---

## 🔧 Installation

```bash
# Create conda environment
conda create -n rapid python==3.11
conda activate rapid

# Install dependencies
cd verl-main
pip install -r requirements.txt

# Install flash-attention (see: https://github.com/Dao-AILab/flash-attention/releases)
# Follow the instructions for your CUDA version

# Install verl-main
pip install -e .

# Install VLMEvalKit
cd ../VLMEvalKit
pip install -e .

pip install transformers==4.51.1
```

---

## 📦 Prepare Training Data

Download [ViRL39K dataset](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and preprocess it:

```bash
python verl-main/examples/data_preprocess/virl39k_pre.py \
  --src-parquet /cache/data/datasets/ViRL39K/39Krelease.parquet \
  --tgt-dir /cache/data/huggingface_datasets/virl39k_hf_no_deepscaler 

python verl-main/examples/data_preprocess/virl39k.py \
  --src-hf-dataset /cache/data/huggingface_datasets/virl39k_hf_no_deepscaler/ \
  --tgt-parquet /cache/data/huggingface_datasets/virl39k_no_deepscaler_caption.parquet

python verl-main/examples/data_preprocess/virl39k_qa.py \
  --src-hf-dataset /cache/data/huggingface_datasets/virl39k_hf_no_deepscaler/ \
  --tgt-parquet /cache/data/huggingface_datasets/virl39k_no_deepscaler_qa.parquet
```

---

## 🏋️‍♂️ Training

### Train Qwen2.5-VL-7B with GRPO

```bash
bash verl-main/examples/grpo_trainer/grpo_7b.sh
```
### Train the resulting ckeckpoint with VPO (using R1-Distilled-7B as the reasoner)
```bash
bash verl-main/examples/grpo_trainer/vpo_7b.sh
```

### Convert Checkpoints to HuggingFace Format
```bash
bash verl-main/scripts/convert2hf.sh
```

## 🔍 Inference and Evaluation

```bash
bash VLMEvalKit/run_inference.sh
```

---

## 📁 Directory Structure

```
.
├── verl-main/
│   ├── examples/
│   ├── scripts/
│   └── ...
└── VLMEvalKit/
    ├── outputs/
    └── ...

```



## 🤝 Acknowledgements
- [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
- [TIGER Lab ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)
- [OmniCaptioner](https://github.com/Alpha-Innovator/OmniCaptioner)