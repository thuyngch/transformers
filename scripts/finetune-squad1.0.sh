#!/usr/bin/env bash
SQUAD_DIR="/home/cybercore/thuync/datasets/SQuAD1.0"
OUT_DIR="/home/member/Workspace/thuync/checkpoints/wwm_uncased_finetuned_squad"
BATCH_SIZE=48

export CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=4

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
	examples/run_squad.py \
	--model_type bert \
	--model_name_or_path bert-base-cased \
	--do_train \
	--do_eval \
	--do_lower_case \
	--train_file "${SQUAD_DIR}/train-v1.1.json" \
	--predict_file "${SQUAD_DIR}/dev-v1.1.json" \
	--learning_rate 3e-5 \
	--num_train_epochs 2 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir ${OUT_DIR} \
	--per_gpu_train_batch_size ${BATCH_SIZE} \
	--gradient_accumulation_steps 12
