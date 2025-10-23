set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_7b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/ma-user/work/cache/data/datasets/deepscaler_40k_r1_1.5b_correct_sample1.parquet \
    data.val_files=/cache/data/datasets/simplelr_qwen_level3to5/test2.parquet \
    data.prompt_key=problem \
    data.response_key=generated_text \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=32768 \
    data.truncation=right \
    model.use_liger=True \
    model.partial_pretrain=/home/ma-user/work/cache/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B \
    trainer.default_local_dir=/home/ma-user/work/cache/exps/checkpoints/ds-r1-1.5b-rft \
    trainer.project_name=ds-r1-1.5b-rft  \
    trainer.experiment_name=ds-r1-1.5b-rft  \
    trainer.total_epochs=1 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true