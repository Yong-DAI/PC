# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, task_key_norm= None,class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()
    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    batch_sum = 0
    idacc_sum = 0 
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
#                 gt1k_cls = output['logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id,task_key_norm= task_key_norm, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
         # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
#         print('logit shape:',logits.shape, 'target shape:', target.shape)
        if args.pull_constraint and 'reduce_sim' in output:
            # print('loss:',loss)   ### 2.26
            # print('loss_DP',output['reduce_sim'])     ### 0.67
            loss = loss #- args.pull_constraint_coeff * output['reduce_sim']
        
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        gt = output['idx_gt'].cpu().numpy()
        pred = output['idx_pred'].cpu().numpy()
        ##print('pred:',pred)
        idacc = float(np.sum(gt == pred))/float(np.size(gt,0))
        ## print('idacc',idacc)
        idacc_sum += idacc
        batch_sum +=1
    # gather the stats from all processes
    pred_vs_gt = 1.0* idacc_sum/batch_sum
        
    # accuracy = float(torch.sum(torch.squeeze(pred) == torch.argmax(all_label, dim=1))) / float(all_label.size()[0])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, 'pred_vs_gt:', pred_vs_gt)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred_vs_gt


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, task_key_norm= None,class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    batch_sum = 0
    idacc_sum = 0 
    
    with torch.no_grad():
#         prompt_all_l0 = torch.zeros(1,50,768).to(device, non_blocking=True)
#         prompt_all_l1 = torch.zeros(1,50,768).to(device, non_blocking=True)
#         prompt_all_l2 = torch.zeros(1,50,768).to(device, non_blocking=True)
#         target_all = torch.zeros(1,).to(device, non_blocking=True)
        
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, task_key_norm= task_key_norm, cls_features=cls_features)
            logits = output['logits']
            
# #             prompt_l0 =torch.mean(output['prompt_l0'], dim=1) 
#             prompt_l0 =output['prompt_l0']
#             prompt_all_l0 = torch.cat([prompt_all_l0,prompt_l0], dim=0)
# #             prompt_l1 = torch.mean(output['prompt_l1'], dim=1) 
#             prompt_l1 =output['prompt_l1']
#             prompt_all_l1 = torch.cat([prompt_all_l1,prompt_l1], dim=0)
# #             prompt_l2 = torch.mean(output['prompt_l2'], dim=1)
#             prompt_l2 =output['prompt_l2']
#             prompt_all_l2 = torch.cat([prompt_all_l2,prompt_l2], dim=0)
#             target_all = torch.cat([target_all,target], dim=0)
                             
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            
#             gt = output['idx_gt'].cpu().numpy()
#             pred = output['idx_pred'].cpu().numpy()
# #             print('pred:',pred[0],'   gt:', gt[0])
#             idacc = float(np.sum(gt == pred))/float(np.size(gt,0))
#             ## print('idacc',idacc)
#             idacc_sum += idacc
#             batch_sum +=1
#         # gather the stats from all processes
#     pred_vs_gt = 1.0* idacc_sum/batch_sum
    pred_vs_gt = 1.0
    
#     torch.save({'gt_feat': prompt_all_l0,'gt_targets': target_all}, 'tsne/cif50/prompt_all_l0_gt_task_%s.pth'%task_id)
#     torch.save({'gt_feat': prompt_all_l1,'gt_targets': target_all}, 'tsne/cif50/prompt_all_l1_gt_task_%s.pth'%task_id)
#     torch.save({'gt_feat': prompt_all_l2,'gt_targets': target_all}, 'tsne/cif50/prompt_all_l2_gt_task_%s.pth'%task_id)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} '
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    print('test task_id:  ',task_id,'  pred_vs_gt:', pred_vs_gt)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred_vs_gt


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1,task_key_norm= None, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    
    vs_gt_sum = 0
    id_sum = 0
    for i in range(task_id+1):
        test_stats, pred_vs_gt = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i,task_key_norm= task_key_norm, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        result_till_task = "[accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}, pred_vs_gt: {:.4f}".format(task_id+1, test_stats['Acc@1'], test_stats['Acc@5'], test_stats['Loss'] ,pred_vs_gt )
        with open(os.path.join(args.output_dir, 'log/log_till_task.txt'), 'a') as f:
            f.write(result_till_task + '\n')
        
        vs_gt_sum +=pred_vs_gt
        id_sum +=1
        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    vs_gt_avg = vs_gt_sum/id_sum
    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}, pred_vs_gt_avg: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2] ,vs_gt_avg )
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    with open(os.path.join(args.output_dir, 'checkpoint/{}_average.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
        f.write(result_str + '\n')

    return test_stats, vs_gt_avg

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
#     eval_t = locals()
    key_shape = (10, 768)
    task_key_norm = torch.zeros(key_shape,dtype = torch.float32)
    
    for task_id in range(args.num_tasks):
        
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
        
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0 :     ##  and task_id!=3 and task_id!=9  #####################################
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
        
            
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        tota_epoch = args.epochs
#         if task_id ==3 or task_id ==9:
#             tota_epoch = 6
        for epoch in range(tota_epoch):            
            train_stats ,pred_vs_gt_tra = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id,task_key_norm= task_key_norm, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
                
#         with torch.no_grad():
#             if args.distributed:
#                 model.module.e_prompt.prompt_key.grad.zero_()
#                 task_key_norm[task_id,:] = model.e_prompt.prompt_key[task_id,:]
#                 optimizer.param_groups[0]['params'] = model.module.parameters()
#             else:
#                 model.e_prompt.prompt_key.grad.zero_()
#                 task_key_norm[task_id,:] = model.e_prompt.prompt_key[task_id,:]
#                 optimizer.param_groups[0]['params'] = model.parameters()
                
        test_stats, pred_vs_gt_te = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id,task_key_norm= task_key_norm, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'pred_vs_gt_tra': pred_vs_gt_tra,
            'pred_vs_gt_te': pred_vs_gt_te}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'checkpoint/{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
        
        
#         eval_t['task_key_norm_task_'+ str(task_id)]  = task_key_norm.cpu().numpy()
        # np.save('./task_key_norm/task_key_norm_task_%s.npy'%task_id, task_key_norm.cpu().numpy())
