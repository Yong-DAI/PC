# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continual_dataloader import ContinualDataLoader
from engine import *
import models
import utils

import torch.optim as optim

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def get_args_parser():
    parser = argparse.ArgumentParser('DualPrompt CIFAR-100 training and evaluation configs', add_help=False)

    parser.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=False, help='scaling lr by batch size (default: False)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='../../dual_prompt/datasets/cifar100/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name')
#     parser.add_argument('--data-path', default='../datasets/ImageNet_R/', type=str, help='dataset path')
#     parser.add_argument('--dataset', default='imagenet_r', type=str, help='dataset name')
    
    parser.add_argument('--shuffle', default=True, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)  ###### 
    parser.add_argument('--eval', default = False, help='Perform evaluation only')                                               ##################
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                       help='number of nodes for distributed training')
#     parser.add_argument('--rank', default=0, type=int,
#                        help='node rank for distributed training')
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument('--dist_url', default='tcp://192.168.202.25:6686', help='url used to set up distributed training')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--classes_per_task', default=10, type=int, help='number of classes per task')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')                             #### mask
    parser.add_argument('--task_id_infer', default=True, type=bool, help='if doing task incremental') 
    
    # G-Prompt parameters
    parser.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
    parser.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
    parser.add_argument('--g_prompt_layer_idx', default=[0, 1], type=int, nargs = "+", help='the layer index of the G-Prompt')
    parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')
    
    # E-Prompt parameters
    parser.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
    parser.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs = "+", help='the layer index of the E-Prompt')
    parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

    # Use prompt pool in L2P to implement E-Prompt
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--size', default=10, type=int,)
    parser.add_argument('--length', default=20,type=int, )
    parser.add_argument('--top_k', default=1, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str,)
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=True, type=bool)
    parser.add_argument('--mask_first_epoch', default=False, type=bool)
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)      ### True
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=1.0, type=float)
    parser.add_argument('--same_key_value', default=False, type=bool)

    # ViT parameters
    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    parser.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    return parser


def main(args):
    print('cuda count:',torch.cuda.device_count())
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    continual_dataloader = ContinualDataLoader(args)
    data_loader, class_mask = continual_dataloader.create_dataloader()

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,   ### args.nb_classes
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,           ### True
        num_classes=args.nb_classes,          ### 100
        drop_rate=args.drop,                   ### 0.0
        drop_path_rate=args.drop_path,          ### 0.0
        drop_block_rate=None,          
        
        prompt_length=args.length,              ###   25
        embedding_key=args.embedding_key,       ###  cls
        prompt_init=args.prompt_key_init,       ### uniform
        prompt_pool=args.prompt_pool,           ### True
        prompt_key=args.prompt_key,             ### True
        pool_size=args.size,                    ### 10
        top_k=args.top_k,                        ### 1
        batchwise_prompt=args.batchwise_prompt,        ### True
        prompt_key_init=args.prompt_key_init,        ### uniform
        head_type=args.head_type,                    ### token
        use_prompt_mask=args.use_prompt_mask,         ### True
        use_g_prompt=args.use_g_prompt,               ### True
        g_prompt_length=args.g_prompt_length,          ### 5
        g_prompt_layer_idx=args.g_prompt_layer_idx,        ### [0,1]
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,      ###True
        use_e_prompt=args.use_e_prompt,                    ### True
        e_prompt_layer_idx=args.e_prompt_layer_idx,         ###[2,3,4]
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,        ### True
        same_key_value=args.same_key_value,               #### False
    )
    original_model.to(device)
    model.to(device)  

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    
    # print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

#         for task_id in range(args.num_tasks):
#             checkpoint_path = os.path.join(args.output_dir, 'R-1maben-3(epgn-wgn12fc)-70.36/task{}_checkpoint.pth'.format(task_id+1))
#             if os.path.exists(checkpoint_path):
#                 print('Loading checkpoint from:', checkpoint_path)
#                 checkpoint = torch.load(checkpoint_path)
#                 model.load_state_dict(checkpoint['model'])
#             else:
#                 print('No checkpoint found at:', checkpoint_path)
#                 return
#             key_shape = (10, 768)
#             task_key_norm = torch.zeros(key_shape,dtype = torch.float32)
#             _ = evaluate_till_now(model, original_model, data_loader, device, 
#                                             task_id, task_key_norm,class_mask, acc_matrix, args)
        
        key_shape = (10, 768)
        task_key_norm = torch.zeros(key_shape,dtype = torch.float32)
        checkpoint_path = os.path.join(args.output_dir, 'cif-87.20/task{}_checkpoint.pth'.format(args.num_tasks))
        print('Loading checkpoint from:', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'],False)
        # _ = evaluate_till_now(model, original_model, data_loader, device, 
        #                                     args.num_tasks-1, task_key_norm,class_mask, acc_matrix, args)
        # return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size * args.world_size
    else:
        global_batch_size = args.batch_size
    args.lr = args.lr * global_batch_size / 256.0

    # optimizer = create_optimizer(args, model_without_ddp)
    
    ##########################
    maben_para = model_without_ddp.e_prompt.maben#.parameters()
    ignored_maben = map(id, model_without_ddp.e_prompt.maben)
    print('ignored_maben',ignored_maben)
    # print('model_without_ddp.parameters():',model_without_ddp.parameters())
    
    base_params = filter(lambda p: id(p) not in ignored_maben, model_without_ddp.parameters())
    print('base_params:',base_params)
    
    optimizer = optim.Adam([{'params': base_params},
                                {'params':maben_para, 'lr':  args.lr*0.2}], lr =  args.lr)
    ############################
    
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    # must pass the criterion to cuda() to make it work
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt CIFAR-100 training and evaluation configs', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    sys.exit(0)
