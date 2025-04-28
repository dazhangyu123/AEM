# !/usr/bin/env python
import sys
import os

import torch.nn.functional as F

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed, MetricLogger, SmoothedValue, \
    adjust_learning_rate
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import ABMIL, MHA
from architecture.dsmil import MILNet, FCLayer, BClassifier

from timm.utils import accuracy
import torchmetrics
import time
import wandb
import scipy.stats as stats
import numpy as np
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/bracs_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument(
        "--seed", type=int, default=2, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument('--arch', default='abmil', choices=['abmil', 'mha', 'dsmil'],
                        help='the MIL method choice')
    parser.add_argument("--lamda", type=float, default=0.1,
                        help='lambda used for balancing cross-entropy loss and rank loss.')
    parser.add_argument("--top_k", default=-1, type=int, help="Top-K instances for using the AEM loss.")
    parser.add_argument('--pretrain', default='conch', choices=['natural_supervised', 'medical_ssl', 'plip', 'path-clip-B-AAAI',
                                                                      'path-clip-B', 'path-clip-L-336', 'openai-clip-B', 'conch',
                                                                     'openai-clip-L-336', 'quilt-net', 'biomedclip', 'UNI', 'GigaPath'],
                        help='pretrain methods')
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="Steps of loss accumulation."
    )
    parser.add_argument("--subsampling", type=float, default=1.0, help='the ratio of subsampling')
    args = parser.parse_args()
    return args

def main():
    # Load config file
    args = get_arguments()

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    conf.lr = conf.lr * conf.accumulation_steps
    if conf.pretrain == 'medical_ssl':
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain == 'natural_supervised':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-B' or conf.pretrain == 'openai-clip-B' or conf.pretrain == 'plip'\
            or conf.pretrain == 'quilt-net'  or conf.pretrain == 'path-clip-B-AAAI'  or conf.pretrain == 'biomedclip' or conf.pretrain == 'conch':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-L-336' or conf.pretrain == 'openai-clip-L-336':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'UNI':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat = 1536
        conf.D_inner = 768


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ADR",
        # track hyperparameters and run metadata
        config={'dataset': conf.dataset,
                'pretrain': conf.pretrain,
                'loss_form': conf.arch,
                'lamda': conf.lamda,
                'subsampling': conf.subsampling,
                'top_k': conf.top_k,
                'seed': conf.seed,},
        mode=conf.wandb_mode
    )
    run_dir = wandb.run.dir  # Get the wandb run directory
    print('Wandb run dir: %s'%run_dir)
    ckpt_dir = os.path.join(os.path.dirname(os.path.normpath(run_dir)), 'saved_models')
    os.makedirs(ckpt_dir, exist_ok=True)  # Create the 'ckpt' directory if it doesn't exist

    print("Used config:");
    pprint(vars(conf));


    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False, persistent_workers=True)

    # define network
    if conf.arch == 'abmil':
        model = ABMIL(conf)
    elif conf.arch == 'dsmil':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        model = MILNet(i_classifier, b_classifier)
    elif conf.arch == 'mha':
        model = MHA(conf)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)
    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)

    # Record the start time
    start_time = time.time()

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):
        train_one_epoch(model, train_loader, optimizer, device, epoch, conf)


        val_auc, val_acc, val_f1, val_loss, val_div_loss = evaluate(model, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss, test_div_loss = evaluate(model, test_loader, device, conf, 'Test')



        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
        print('\n')

    save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
               save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))
    print("Results on best epoch:")
    print(best_state)

    # Calculate the total training time
    training_time_seconds = time.time() - start_time

    # Print the total training time
    print(f"Total training time: {training_time_seconds} seconds")
    wandb.finish()

def train_one_epoch(model, data_loader, optimizer, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # 添加梯度累积相关参数
    optimizer.zero_grad()   # 在epoch开始时清零梯度

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)

        if conf.subsampling < 1.0:
            # Calculate the number of samples (80% of 100)
            num_samples = int(conf.subsampling * image_patches.shape[1])
            # Generate random permutation of indices
            indices = torch.randperm(image_patches.shape[1])
            # Select the first 80% of the permuted indices
            sampled_indices = indices[:num_samples].to(device)
            # Use the sampled indices to select rows from the tensor
            image_patches = image_patches[:,sampled_indices]

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)

        # Compute loss
        if conf.arch == 'dsmil':
            ins_preds, bag_preds, attn = model(image_patches)
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            bag_loss = 0.5 * criterion(max_preds, labels) + 0.5 * criterion(bag_preds, labels)
        else:
            bag_logit, attn = model(image_patches, is_train=True)
            bag_loss = criterion(bag_logit, labels)

        if conf.arch == 'mha' or conf.arch == 'dsmil':
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[0]
        else:
            if conf.top_k > 0:
                A_softmax = torch.softmax(attn, dim=-1)
                values, _ = torch.topk(A_softmax, min(conf.top_k, A_softmax.size(-1)), dim=-1)
                values_sum = values.sum(dim=-1, keepdim=True)
                values_normalized = values / values_sum
                div_loss = torch.sum(values_normalized * torch.log(values_normalized))
            else:
                div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1))



        # Compute weight
        loss = conf.lamda * div_loss + bag_loss

        # 将损失除以累积步数
        loss = loss / conf.accumulation_steps
        loss.backward()

        # 每accumulation_steps步进行一次参数更新
        if (data_it + 1) % conf.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(slide_loss=bag_loss.item())
        metric_logger.update(div_loss=div_loss.item())

        if conf.wandb_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb.log({'div_loss': div_loss}, commit=False)
            wandb.log({'bag_loss': bag_loss})

    # 处理最后不足accumulation_steps的batch
    if (data_it + 1) % conf.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, conf, header):

    # Set the network to evaluation mode
    model.eval()

    y_pred = []
    y_true = []
    attention_entropies = []
    validation_losses = []

    metric_logger = MetricLogger(delimiter="  ")
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        label = data['label'].to(device)

        if conf.arch == 'dsmil':
            instance_logits, bag_logit, attn = model(image_patches)
            max_preds, _ = torch.max(instance_logits, 0, keepdim=True)
            loss = 0.5 * criterion(bag_logit, label) \
                   + 0.5 * criterion(max_preds, label)
            pred = 0.5 * torch.softmax(max_preds, dim=-1) \
                   + 0.5 * torch.softmax(bag_logit, dim=-1)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[0]
        else:
            bag_logit, attn = model(image_patches)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / (attn.shape[0] * attn.shape[1])
            loss = criterion(bag_logit, label)
            pred = torch.softmax(bag_logit, dim=-1)

        # 记录每个batch的loss和entropy
        attention_entropies.append(div_loss.item())
        validation_losses.append(loss.item())
        acc1 = accuracy(pred, label, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.update(div_loss = div_loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # 计算Pearson相关系数
    correlation, p_value = stats.pearsonr(attention_entropies, validation_losses)

    AUROC_metric = torchmetrics.AUROC(task="multiclass", num_classes = conf.n_class, average = 'macro').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(task="multiclass", num_classes = conf.n_class, average = 'macro').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()
    # AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, average = 'macro').to(device)
    # AUROC_metric(y_pred, y_true)
    # auroc = AUROC_metric.compute().item()
    # F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, average = 'macro').to(device)
    # F1_metric(y_pred, y_true)
    # f1_score = F1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} div_loss {div_losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f} Correlation {correlation: .3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, div_losses=metric_logger.div_loss, AUROC=auroc, F1=f1_score, correlation=correlation))

    if conf.wandb_mode != 'disabled':
        wandb.log({'%s/acc1'%header: metric_logger.acc1.global_avg}, commit=False)
        wandb.log({'%s/auc'%header: auroc}, commit=False)
        wandb.log({'%s/f1'%header: f1_score}, commit=False)
        wandb.log({'%s/loss'%header: metric_logger.loss.global_avg}, commit=False)
        wandb.log({'%s/div_loss'%header: metric_logger.div_loss.global_avg}, commit=False)
        wandb.log({'%s/pearson_correlation' % header: correlation}, commit=False)

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, metric_logger.div_loss.global_avg


if __name__ == '__main__':
    main()

