#!/bin/bash

TASK="RTE"
BERT_DIR="model/RTE/bert/"
DATA_DIR="data/RTE"
W_BERT="dynaw_bert/RTE/"
WD_BERT="dynawd_bert/RTE/"

echo $TASK 
echo "dynaw training"
python run_glue.py \
	--model_type bert \
	--task_name $TASK \
	--do_train \
	--data_dir $DATA_DIR \
	--model_dir $BERT_DIR \
	--output_dir $W_BERT \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 8 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 1.0 \
	--width_lambda1 1.0 \
	--width_lambda2 0.1 \
	--logging_steps 200 \
	--training_phase dynabertw \
	--graft_during_training 
	# --data_aug \

echo "dynawd training"
python run_glue.py \
	--model_type bert \
	--task_name $TASK \
	--do_train \
	--data_dir $DATA_DIR \
	--model_dir $W_BERT \
	--output_dir $WD_BERT \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 8 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--depth_lambda1 1.0 \
	--depth_lambda2 1.0 \
	--logging_steps 200 \
	--training_phase dynabert  
	# --data_aug \

echo "final fine-tuning"
python run_glue.py \
	--model_type bert \
	--task_name $TASK \
	--do_train \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT \
	--output_dir $WD_BERT \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--logging_steps 20 \
	--training_phase final_finetuning 