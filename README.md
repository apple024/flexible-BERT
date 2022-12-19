Flexible BERT with width- and depth-dynamic inference
=====================================================
This is the code for paper "Flexible BERT with width- and depth-dynamic inference". The inference framework is built on BERT with optimized subnets, and achieves width- and depth-dynamic inference regarding different input, yielding reduced computation and faster inference on NLU tasks. 

## Installation
Run command below to install the environment
```bash
pip install -r requirements.txt
```

## Train a BERT model with width- and depth-adaptive subnets
Our codes are based on [DynaBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT), including three steps: width-adaptive training, depth-adaptive training, and final fine-tuning. The differences are: (1) we apply Neural grafting technique at the first stage to boost the subnets; (2) we incremantally pick up BERT layers for depth-adaptive traning.

Preparation: (1) download Pretrained task-specific BERT models from [huggingface](https://github.com/huggingface/transformers) and put them in the folder `$BERT_DIR`; (2) generate augmented data using [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) and put them in the folder `$DATA_DIR`.

Run script `run_glue.sh` where RTE data set is taken as an example. The final BERT model with Neural grafting boosted subnets is saved in `$WD_BERT_DIR`.  The script includes three-step training below.

(1) Width-adaptive training.

```
# --graft_during_training, only valid during width-adaptive training.

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
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 1.0 \
	--width_lambda1 1.0 \
	--width_lambda2 0.1 \
	--logging_steps 200 \
	--training_phase dynabertw \
	--graft_during_training \
	--data_aug 
```

(2) Depth-adaptive training.

```
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
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--depth_lambda1 1.0 \
	--depth_lambda2 1.0 \
	--logging_steps 200 \
	--training_phase dynabert  \
	--data_aug \
```

(3) Final fine-tuning without data augmentation. this is optional.

```
python run_glue.py \
	--model_type bert \
	--task_name $TASK \
	--do_train \
	--data_dir $DATA_DIR \
	--model_dir $WD_BERT \
	--output_dir $WD_BERT \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32\
	--per_gpu_eval_batch_size 32\
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--logging_steps 20 \
	--training_phase final_finetuning \
```

## Train gating modules enabling width-dynamic inference
Run script `train_glue.sh` to train tha gating modules. 

```
# --gate_setting, aligning with --width_mult_list, since gating modules are based on the optimized subnets.

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
```

## Dynamic inference
We offer differen ways of inference: static inference using specific subnets, width-dynamic and both with- and depth-dynamic inference. Run script `gate_inference.sh`. 

(1) static inference on specific subnets.

```
# --static, static inference. 
# --width_mul_list, offering four identical width ratios. --depth_mult_list, specifying the depth ratio.

python3 train_gate.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA \
	--model_dir $MODEL \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.5,0.5,0.5,0.5 \
	--depth_mult_list 1.0 \
	--do_eval \
	--record_gpu_time \
	--static \
```

(2) width-dynamic inference, on the whole model(depth=1.0).

```
# --w_dynamic, width-dynamic inference. --width_mult_list, the width ratios for inference.

python3 train_gate_new.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA \
	--model_dir $MODEL \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.5,0.75,0.75,0.75 \
	--depth_mult_list 1.0 \
	--do_eval \
	--record_gpu_time \
	--w_dynamic \
```

(3) width- and depth-dynamic inference.

```
--wd_dynamic, width- and depth-dynamic. 
--early_exit, depth-dynamic inference. --exit_threshold, exit threshold.

python3 train_gate_new.py \
	--model_type bert \
	--task_name $TASK \
	--data_dir $DATA \
	--model_dir $MODEL \
	--output_dir output/ \
	--max_seq_length 128 \
	--per_gpu_eval_batch_size 1 \
	--width_mult_list 0.25,0.25,0.25,0.5 \
	--depth_mult_list 1.0 \
	--do_eval \
	--record_gpu_time \
	--wd_dynamic \
	--early_exit \
	--exit_threshold 0.15 \

```

We also offer the BERT models enabling dynamic inference framework we trained on GLUE benchmark [here](https://nextcloud.hpi.de/s/F9azocsAB3gcm4c).
The config of the best tradeoff inference results on GLUE tasks is in `best_tradeoff_inference_config.json`.