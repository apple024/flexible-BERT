#!/bin/sh

TASK="RTE"
DATA_DIR="data/RTE"
WD_BERT_GATES="dynawd_bert_gates/RTE/"

echo $TASK 
echo "inference"

echo "static"
python3 train_gate.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT_GATES \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.5,0.5,0.5,0.5 \
	--depth_mult_list 0.5 \
	--do_eval \
	--record_gpu_time \
	--static \

echo "width-dynamic"
python3 train_gate.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT_GATES \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.5,0.75,0.75,0.75 \
	--depth_mult_list 1.0 \
	--do_eval \
	--record_gpu_time \
	--w_dynamic \

echo "width- and depth-dynamic"

python3 train_gate.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT_GATES \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.25,0.25,0.5,0.5 \
	--depth_mult_list 1.0 \
	--do_eval \
	--record_gpu_time \
	--wd_dynamic \
	--early_exit \
	--exit_threshold 0.06 \
