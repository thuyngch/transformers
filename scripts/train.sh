#!/usr/bin/env bash
SQUAD_DIR="/home/member/Workspace/thuync/datasets/SQuAD2.0"
OUT_DIR="/home/member/Workspace/thuync/checkpoints/bert_squad2"

MODEL="bert"
MODEL_NAME="bert-base-cased"
BATCH_SIZE=24
LR=3e-5
EPOCHS=25
MAX_LEN=384
STRIDE=128
GRAD_ACCU=12
WARMUP=100
WDECAY=1e-6

export CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=2

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
	tools/squad/train.py \
	--model_type ${MODEL} \
	--model_name_or_path ${MODEL_NAME} \
	--do_train \
	--do_eval \
	--do_lower_case \
	--train_file "${SQUAD_DIR}/train-v2.0.json" \
	--predict_file "${SQUAD_DIR}/dev-v2.0.json" \
	--learning_rate ${LR} \
	--num_train_epochs ${EPOCHS} \
	--max_seq_length ${MAX_LEN} \
	--doc_stride ${STRIDE} \
	--output_dir ${OUT_DIR} \
	--per_gpu_train_batch_size ${BATCH_SIZE} \
	--gradient_accumulation_steps ${GRAD_ACCU} \
	--warmup_steps ${WARMUP} \
	--weight_decay ${WDECAY}
