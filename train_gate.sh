#!/bin/sh

TASK="RTE"
DATA_DIR="data/RTE"
WD_BERT="dynawd_bert/RTE/"
WD_BERT_GATES="dynawd_bert_gates/RTE/"

echo $TASK 
echo "gates training"

python train_gate.py \
	--model_type bert \
	--task_name $TASK \
	--do_train \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT \
	--output_dir $WD_BERT_GATES \
	--max_seq_length 128 \
	--learning_rate 2e-4 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 8 \
	--logging_steps 40 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--gate_setting 0.25,0.5,0.75,1.0 \
	--training_phase traingate \
