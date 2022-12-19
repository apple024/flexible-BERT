# coding=utf-8
# 2020.08.28 - Changed regular fine-tuning to fine-tuning with adaptive width and depth
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors,  the HuggingFace Inc.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import math
import numpy as np
import torch
import time
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from torch.nn import  MSELoss
import torch.nn as nn
import pdb
import thop
# from thop import profile
from profile import *
import csv

from transformers import (BertConfig, BertDsForSequenceClassification, BertTokenizer, BertGate,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def add_flops(model, input_tuple, train=True):
    # UPDATE add param
    flops, params = profile(model, inputs=input_tuple, train=train)
    return flops


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


loss_mse = MSELoss()
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertDsForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, num_gates=12):
    """ Train the model """

    if num_gates not in [1,2,3,4,6,12]:
        num_gates = 12

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type == 'roberta':
        args.warmup_steps = int(t_total*0.06)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'gate' in name:
            param.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    loss_fn = torch.nn.NLLLoss(reduction='none')

    # ground truth gate distribution is unbalanced. We use weights in the loss to handle it.
    # The weights are obtained by preprocessing using the DynaBERT model obtained previously.
    if args.task_name == "cola":
        loss_weight=torch.Tensor([0.3, 14.0, 45.5, 7.5]).to(args.device)
    elif args.task_name == "rte":
        loss_weight=torch.Tensor([0.3, 28.3, 155.6, 12.2]).to(args.device)  
    elif args.task_name == "sst-2":
        loss_weight=torch.Tensor([0.3, 129.5, 495.2, 18.7]).to(args.device)
    elif args.task_name == "mrpc":
        loss_weight=torch.Tensor([0.3, 25.5, 101.9, 11.6]).to(args.device)
    elif args.task_name == "mnli":
        loss_weight=torch.Tensor([0.3, 11.9, 35.6, 5.1]).to(args.device)
    elif args.task_name == "qnli":
        loss_weight=torch.Tensor([0.3, 30.7, 111.4, 16.2]).to(args.device)
    elif args.task_name == "qqp":
        loss_weight=torch.Tensor([0.3, 59.9, 249.9, 52.1]).to(args.device)

    # annealing rate
    annealing_lr = -np.log(0.3)/(len(train_dataset)/args.per_gpu_eval_batch_size*args.num_train_epochs) # cola, rte, mrpc: 8 epochs, others: 3 epochs

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    current_loss = 100.
    temperature_curr = 1.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            if global_step > 0 and global_step % 500 == 0:
                temperature_curr = max(0.5, np.exp(-annealing_lr * global_step))

            # accumulate grads for all sub-networks， there should not be changes in depth
            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', 1.0))
                # select teacher model layers for matching
                if args.training_phase == 'traingate':
                    model = model.module if hasattr(model, 'module') else model
                    n_layers = 12
                    depth = round(depth_mult * n_layers)

                # obtain ground truth gate labels
                setattr(model.bert.encoder, 'mode', 'static')
                setattr(model.bert, 'mode', 'static')
                with torch.no_grad():
                    gate_target = [torch.LongTensor([len(args.gate_setting[:-1])]) for idx in range(batch[3].size(0))] # all 1.0 -->3 

                    for index in range(len(args.gate_setting[:-1])): # for 0.25, 0.5, 0.75
                        setting = args.gate_setting[index]
                        alloutputs, _, _ = model(gate=setting, input_ids=batch[0], attention_mask=batch[1], labels=batch[3], token_type_ids=batch[2], fixed_gate=setting)

                        # classification task
                        acc = torch.argmax(alloutputs[1], 1) - batch[3]

                        for idx in range(acc.size(0)):

                            # # classification task
                            if acc[idx] == 0 and gate_target[idx].item() == len(args.gate_setting[:-1]):  # never been changed
                                gate_target[idx] = torch.LongTensor([index])

                    gate_target = torch.stack(gate_target).squeeze(-1)
                    gate_target = gate_target.view(1, -1).repeat(num_gates, 1).view(-1).to(args.device)

                # train gating modules
                setattr(model.bert.encoder, 'mode', 'dynamic')
                setattr(model.bert, 'mode', 'dynamic')
                setattr(model.bert.encoder, 'stage', 'train')
                setattr(model.bert, 'stage', 'train')
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    # widest
                    alloutputs, all_gates, _ = model(gate=width_mult, tau=temperature_curr, input_ids=batch[0], attention_mask=batch[1],
                                                         labels=batch[3], token_type_ids=batch[2])

                    #  SGS Loss
                    all_gates = torch.stack(all_gates).view(-1, len(args.gate_setting)).to(args.device)
                    gate_loss = loss_fn(torch.log(torch.clamp(all_gates, min=1e-6)), gate_target)

                    if width_mult == 0.25:
                        gate_mask = (gate_target==0).float().view(-1).to(args.device)
                    elif width_mult == 0.5:
                        gate_mask = (gate_target==1).float().view(-1).to(args.device)
                    elif width_mult == 0.75:
                        gate_mask = (gate_target==2).float().view(-1).to(args.device)
                    else:
                        gate_mask = (gate_target==3).float().view(-1).to(args.device)   

                    gate_loss = torch.mean(gate_loss * gate_mask)

                    # gate_mask = gate_mask.view(1, -1).repeat(2, 1).view(-1).to(args.device)
                    # gate_loss = loss_fn(all_gates, gate_target)
                    # gate_loss = torch.mean(gate_loss * gate_mask)

                    #  MAdds Loss
                    input_tuple = (width_mult, temperature_curr, batch[0], batch[1], batch[2], None, None, batch[3])
                    # UPDATE add param
                    running_flops = add_flops(model, input_tuple, train=True)
                    running_flops /= batch[0].size(0)
                    if isinstance(running_flops, torch.Tensor):
                        running_flops = running_flops.float().mean().to(args.device)
                    else:
                        running_flops = torch.FloatTensor([running_flops]).to(args.device)
                    # flops_loss = (running_flops / 1e9) ** 2
                    flops_loss = (running_flops / 1e10) ** 2

                    
                    if width_mult == 0.25:
                        loss = gate_loss * loss_weight[0] #/loss_weight.sum() # + 0.5 * flops_loss + alloutputs[0]  # + 0.5 * flops_loss
                    elif width_mult == 0.5:
                        loss = gate_loss * loss_weight[1] #/loss_weight.sum()
                    elif width_mult == 0.75:
                        loss = gate_loss * loss_weight[2] #/loss_weight.sum()
                    elif width_mult == 1.0:
                        loss = gate_loss * loss_weight[3]  #/loss_weight.sum()

                    if args.n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()

                    loss.backward()

            # clip the accumulated grad from all widths
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # evaluate
                if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    if args.evaluate_during_training:

                        _, gate_loss, _ = evaluate_during_training(args, model, temperature_curr, tokenizer, eval=True, num_gates=num_gates)

                        # save model
                        if gate_loss < current_loss:
                            current_loss = gate_loss

                            file_out_dir_best = args.output_dir
                            logger.info("Saving model checkpoint to %s", file_out_dir_best)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(file_out_dir_best)
                            torch.save(args, os.path.join(file_out_dir_best, 'training_args.bin'))
                            model_to_save.config.to_json_file(os.path.join(file_out_dir_best, CONFIG_NAME))
                            tokenizer.save_vocabulary(file_out_dir_best)


            if 0 < t_total < global_step:
                epoch_iterator.close()
                break

        if 0 < t_total < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate_during_training(args, model, tau, tokenizer, prefix="", gate=1.0, eval=True, num_gates=12):
    """ Evaluate the model """

    if num_gates not in [1,2,3,4,6,12]:
        num_gates = 12

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=eval)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        loss_fn = torch.nn.NLLLoss(reduction='none')

        if args.task_name == "cola":
            loss_weight = torch.Tensor([0.3, 14.0, 45.5, 7.5]).to(args.device)
        elif args.task_name == "rte":
            loss_weight = torch.Tensor([0.3, 28.3, 155.6, 12.2]).to(args.device)  
        elif args.task_name == "sst-2":
            loss_weight=torch.Tensor([0.3, 129.5, 495.2, 18.7]).to(args.device)  
        elif args.task_name == "mrpc":
            loss_weight = torch.Tensor([0.3, 25.5, 101.9, 11.6]).to(args.device)  
        elif args.task_name == "mnli":
            loss_weight = torch.Tensor([0.3, 11.9, 35.6, 5.1]).to(args.device)
        elif args.task_name == "qnli":
            loss_weight=torch.Tensor([0.3, 30.7, 111.4, 16.2]).to(args.device)
        elif args.task_name == "qqp":
            loss_weight=torch.Tensor([0.3, 59.9, 249.9, 52.1]).to(args.device)


        eval_loss_all = [0.0] * len(args.gate_setting)
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        flops_loss_all = [0.0] * len(args.gate_setting)
        gate_loss_all = [0.0] * len(args.gate_setting)
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            setattr(model.bert.encoder, 'mode', 'static')
            setattr(model.bert, 'mode', 'static')
            with torch.no_grad():
                gate_target = [torch.LongTensor([len(args.gate_setting[:-1])]) for idx in range(batch[3].size(0))]

                for index in range(len(args.gate_setting[:-1])):
                    setting = args.gate_setting[index]
                    alloutputs, _, _ = model(gate=setting, tau=1.0, input_ids=batch[0], attention_mask=batch[1], labels=batch[3], token_type_ids=batch[2], fixed_gate=setting)
                    
                    # # regression task
                    # acc = alloutputs[1].view(-1) - batch[3] 

                    acc = torch.argmax(alloutputs[1], 1) - batch[3]

                    for idx in range(acc.size(0)):

                        # # regression task
                        # if acc[idx] < args.regression_threshold and gate_target[idx].item() == len(args.gate_setting[:-1]):
                        #     gate_target[idx] = torch.LongTensor([index])

                        if acc[idx] == 0 and gate_target[idx].item() == len(args.gate_setting[:-1]):  # never been changed
                            gate_target[idx] = torch.LongTensor([index])

                gate_target = torch.stack(gate_target).squeeze(-1)
                gate_target = gate_target.view(1, -1).repeat(num_gates, 1).view(-1).to(args.device)

                setattr(model.bert.encoder, 'mode', 'dynamic')
                setattr(model.bert, 'mode', 'dynamic')
                setattr(model.bert.encoder, 'stage', 'train')
                setattr(model.bert, 'stage', 'train')

                count_width = 0
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    # print(f'Count width: {count_width}')
                    inputs = {'input_ids': batch[0], 'tau': tau, 'gate': width_mult, 'attention_mask': batch[1], 'labels': batch[3], 'fixed_gate': args.fixed_gate}
                    if args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None
                    outputs, all_gates, _ = model(**inputs)

                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss_all[count_width] += tmp_eval_loss.mean().item()

                    #  MAdds Loss
                    input_tuple = (width_mult, tau, batch[0], batch[1], batch[2], None, None, batch[3])
                    # UPDATE add param
                    running_flops = add_flops(model, input_tuple, train=False)
                    running_flops /= batch[0].size(0)  # average on each sample
                    if isinstance(running_flops, torch.Tensor):
                        running_flops = running_flops.float().mean()
                    else:
                        running_flops = torch.FloatTensor([running_flops])
                    # flops_loss = (running_flops / 1e9) ** 2
                    flops_loss_all[count_width] += (running_flops.item() / 1e10)

                    all_gates = torch.stack(all_gates).view(-1, len(args.gate_setting)).to(args.device)  # 12, 32, 4
                    gate_loss = loss_fn(torch.log(torch.clamp(all_gates, min=1e-6)), gate_target)

                    # with mask
                    if width_mult == 0.25:
                        gate_mask = (gate_target==0).float().view(-1).to(args.device)
                    elif width_mult == 0.5:
                        gate_mask = (gate_target==1).float().view(-1).to(args.device)
                    elif width_mult == 0.75:
                        gate_mask = (gate_target==2).float().view(-1).to(args.device)
                    else:
                        gate_mask = (gate_target==3).float().view(-1).to(args.device)   

                    gate_loss = torch.mean(gate_loss * gate_mask)

                    if width_mult == 0.25:
                        gate_loss_all[count_width] += gate_loss.item() * loss_weight[0].item()  # / loss_weight.sum().item() # + 0.5 * flops_loss + alloutputs[0]  # + 0.5 * flops_loss
                    elif width_mult == 0.5:
                        gate_loss_all[count_width] += gate_loss.item() * loss_weight[1].item()  # / loss_weight.sum().item()
                    elif width_mult == 0.75:
                        gate_loss_all[count_width] += gate_loss.item() * loss_weight[2].item()  # / loss_weight.sum().item()
                    elif width_mult == 1.0:
                        gate_loss_all[count_width] += gate_loss.item() * loss_weight[3].item()  # / loss_weight.sum().item()

                    count_width += 1

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results.update({'acc_mm': result['acc']})
        else:
            results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")  # wirte all the results to the same file
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")

    avg_gate_loss = np.mean(gate_loss_all) / nb_eval_steps
    avg_flops_loss = np.mean(flops_loss_all) / nb_eval_steps
    return results, avg_gate_loss, avg_flops_loss


#UPDATE new param
def inference(args, model, tokenizer, prefix="", gate=1.0, fixed_gate=None, num_gates=12):
    """ Evaluate the model """

    if num_gates not in [1,2,3,4,6,12]:
        num_gates = 12

    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)

    eval_task_names = (args.task_name,)

    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # the hardness ditribution of the samples are imbalanced and we obtain the weight by preprocessing.
        if args.task_name == "cola":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 14.0, 45.5, 7.5]).to(args.device))
        elif args.task_name == "rte":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 28.3, 155.6, 12.2]).to(args.device))
        elif args.task_name == "sst-2":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 129.5, 495.2, 18.7]).to(args.device))     
        elif args.task_name == "mrpc":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 25.5, 101.9, 11.6]).to(args.device)) 
        elif args.task_name == "mnli":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 30.7, 111.4, 16.2]).to(args.device))
        elif args.task_name == "qnli":
            loss_fn = torch.nn.NLLLoss(weight=torch.Tensor([0.3, 25.5, 101.9, 11.6]).to(args.device))
        else:
            loss_fn = torch.nn.NLLLoss().to(args.device)

        preds = None
        out_label_ids = None
        nb_eval_steps = 0
        flops_loss = 0.
        eval_loss = 0.0

        if args.record_gpu_time:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_timings = np.zeros((len(eval_dataloader), 1))
        elif args.record_cpu_time:
            cpu_timings = np.zeros((len(eval_dataloader), 1))

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():

                setattr(model.bert.encoder, 'mode', 'dynamic')
                setattr(model.bert, 'mode', 'dynamic')
                setattr(model.bert.encoder, 'stage', 'inference')
                setattr(model.bert, 'stage', 'inference')

                width_mult = 0.  # just offer an initial value, does not indicate any meaning.
                inputs = {'input_ids': batch[0], 'gate': width_mult, 'attention_mask': batch[1], 'labels': batch[3], 'fixed_gate': fixed_gate}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None

                if args.record_cpu_time:
                    cpu_starttime = time.time()
                    outputs, all_gates, _ = model(**inputs)
                    cpu_endtime = time.time()
                    cpu_timings[nb_eval_steps] = cpu_endtime - cpu_starttime
                elif args.record_gpu_time:
                    starter.record()
                    outputs, all_gates, _ = model(**inputs)
                    ender.record()
                    torch.cuda.synchronize()
                    gpu_timings[nb_eval_steps] = starter.elapsed_time(ender)
                else:
                    outputs, all_gates, _ = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

                #  MAdds Loss
                input_tuple = (width_mult, batch[0], batch[1], batch[2], None, None, batch[3])
                # UPDATE add param
                running_flops = add_flops(model, input_tuple, train=False)
                running_flops /= batch[0].size(0)
                if isinstance(running_flops, torch.Tensor):
                    running_flops = running_flops.float().mean().to(args.device)
                else:
                    running_flops = torch.FloatTensor([running_flops]).to(args.device)

                flops_loss += running_flops.mean().item() / 1e10 * 2 

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.record_cpu_time:
             results.update({'cpu_latency': np.mean(cpu_timings)})
        elif args.record_gpu_time:
             results.update({'gpu_latency': np.mean(gpu_timings)})

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results.update({'acc_mm': result['acc']})
        else:
            results.update(result)

        flops_loss /= nb_eval_steps
        eval_loss /= nb_eval_steps

        results.update({'flops': flops_loss})
        results.update({'eval_loss': eval_loss})

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # evaluate=False
    processor = processors[task]()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        label_list[1], label_list[2] = label_list[2], label_list[1]
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    if not evaluate and args.data_aug:
        examples_aug = processor.get_train_examples_aug(args.data_dir)
        examples = examples + examples_aug

    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def compute_neuron_head_importance(args, model, tokenizer):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # collect weights
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    model.to(args.device)

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, _, label_ids = batch
            segment_ids = batch[2] if args.model_type == 'bert' else None  # RoBERTa does't use segment_ids

            # calculate head importance
            outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                            head_mask=head_mask)
            loss = outputs[0]
            loss.backward()
            head_importance += head_mask.grad.abs().detach()

            # calculate  neuron importance
            for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
                current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
                current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    """ reorder neurons based on their importance.

        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The student (and teacher) model dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where trained model is saved.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation.")
    parser.add_argument("--evaluate_during_training", default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=200,
                        help="random seed for initialization")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="dropout rate on hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="dropout rate on attention probs.")

    parser.add_argument('--data_aug', action='store_true', help="whether using data augmentation")
    # for depth direction
    parser.add_argument('--depth_mult_list', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument("--depth_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--depth_lambda2", default=1.0, type=float,
                        help="hidden states matching coef.")
    # for width direction
    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="the possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training. Also sets the gate setting.")
    parser.add_argument("--width_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--width_lambda2", default=0.1, type=float,
                        help="hidden states matching coef.")

    parser.add_argument('--gate_setting', type=str, default='1.',
                        help="the possible widths used for gate training, e.g., '1.' is for default")

    parser.add_argument("--training_phase", default="dynabertw", type=str,
                        help="can be finetuning, dynabertw, dynabert, final_finetuning")

    parser.add_argument("--fixed_gate", default=None, type=float,
                        help="the width used for inference")
    parser.add_argument("--num_gates", default=12, type=int,
                        help="the number of gates used")

    parser.add_argument('--record_gpu_time', action='store_true', help="whether to record the GPU inference timing")
    parser.add_argument('--record_cpu_time', action='store_true', help="whether to record the CPU inference timing")

    parser.add_argument('--early_exit', action='store_true', help="whether to use early exit")
    parser.add_argument("--exit_threshold", default=0.57, type=float, help="exit threshold")
    parser.add_argument("--regression_threshold", default=0.5, type=float,
                        help="hidden states matching coef.")
    parser.add_argument('--off_ramps', type=str, default='5,8',
                        help="The indices after which off ramps are placeed")
    parser.add_argument('--cpu', action='store_true', help="whether to run on cpu even if cuda is availible")

    parser.add_argument('--static', action='store_true', help="conduct static inference")
    parser.add_argument('--w_dynamic', action='store_true', help="conduct width-dynamic inference")
    parser.add_argument('--d_dynamic', action='store_true', help="conduct depth-dynamic inference")
    parser.add_argument('--wd_dynamic', action='store_true', help="conduct width- and depth-dynamic inference")

    args = parser.parse_args()

    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]
    args.gate_setting = [float(gate) for gate in args.gate_setting.split(',')]
    args.off_ramps = [int(off_ramps) for off_ramps in args.off_ramps.split(',')]

    # Setup CUDA, GPU & distributed training
    if not args.cpu:
        device = "cuda" if torch.cuda.is_available()  else "cpu"
    else:
        device = "cpu"

    # args.device = device
    # args.n_gpu = torch.cuda.device_count()

    args.device = device
    args.n_gpu = 1

    # Set seed
    set_seed(args)

    # Prepare GLUE task: provide num_labels here
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # prepare model, tokernizer and config
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    config.output_attentions, config.output_hidden_states, config.output_intermediate = True, True, True
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)

    # add gate settings
    config.gate_setting = args.width_mult_list
    config.num_gates = args.num_gates
    config.early_exit = args.early_exit
    config.exit_threshold = args.exit_threshold
    config.off_ramps = args.off_ramps
    config.device = args.device

    # load student model if necessary
    model = model_class.from_pretrained(args.model_dir, config=config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.early_exit:
        dict_dest = model.state_dict()
        for name, param in dict_dest.items():
            if 'off_ramps' in name and 'weight' in name:
                dict_dest[name].data.copy_(dict_dest['classifier.weight'])
            if 'off_ramps' in name and 'bias' in name:
                dict_dest[name].data.copy_(dict_dest['classifier.bias'])
            if 'poolers' in name and 'weight' in name:
                dict_dest[name].data.copy_(dict_dest['bert.pooler.dense.weight'])
            if 'poolers' in name and 'bias' in name:
                dict_dest[name].data.copy_(dict_dest['bert.pooler.dense.bias'])

        model.load_state_dict(dict_dest)

    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, num_gates=args.num_gates)

    elif args.do_eval:
        acc = []
        if args.task_name == "mnli":   # for both MNLI-m and MNLI-mm
            acc_both = []

        if args.static:  # static inference
            if len(args.width_mult_list) == 1 or len(list(set(args.width_mult_list))) == 1 or len(args.depth_mult_list) > 1:
                logger.error("set the width and depth ratio of the subnet for static inference.")
                logger.error("provide four width ratios in the same value and one depth ratio.")

            for width_mult in sorted(args.width_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))

        if args.w_dynamic:  # width-dynamic inference
            if len(args.depth_mult_list) > 1:
                logger.error("set one depth ratio of the deepest subnet for width dynamic inference.")
                logger.error("set it to be 1.0 to conduct width-dynamic inference on BERT model.")

            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))

        if args.d_dynamic:  # depth- dynamic inference, i.e., early exit
            if len(args.depth_mult_list) > 1:
                logger.error("set it to be 1.0 to conduct depth-dynamic inference on BERT model.")
            if len(args.width_mult_list) == 1 or len(list(set(args.width_mult_list))) == 1:
                logger.error("set the width ratio of the widest subnet for depth-dynamic inference.")
                logger.error("provide four width ratios in the same value.")

            if args.early_exit is None:
                logger.error("set early exit to be true.")  # --early_exit
                logger.error("set the early exit threshold.") # --exit_threshold

            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))

            
        if args.wd_dynamic:  # width- and depth- dynamic inference
            if len(args.depth_mult_list) > 1:
                logger.error("set it to be 1.0 to conduct width- and depth-dynamic inference on BERT model.")
            if args.early_exit is None:
                logger.error("set early exit to be true.")  # --early_exit
                logger.error("set the early exit threshold.") # --exit_threshold

            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))


        results = inference(args, model, tokenizer, prefix="", gate=0., fixed_gate=args.fixed_gate, num_gates=args.num_gates)

        print("***** Eval results: %s *****"%(str(args.task_name)))
        if args.static:
            print("static subnet inference.")
            print(" depth ratio: %s" % (str(args.depth_mult_list)))
        if args.w_dynamic:
            print("width-dynamic inference.")
            print(" depth ratio: %s" % (str(args.depth_mult_list)))
        if args.wd_dynamic:
            print("width- and depth-dynamic inference.")
            print(" early exit threshold: %s" % (str(args.exit_threshold)))
        print(" width ratios: %s" % (str(args.width_mult_list)))
        for key in sorted(results.keys()):
            print(" %s = %s" % (key, results[key]))


if __name__ == "__main__":
    main()
