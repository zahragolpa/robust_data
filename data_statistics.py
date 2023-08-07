# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : Data_Statistics.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 18:56
@Author   : Zhiheng Xi
"""

import argparse
import logging
import os
from pathlib import Path
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import cv2
import time
from transformations import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import (
    AdamW, AutoConfig, AutoTokenizer, AutoFeatureExtractor, ViTForImageClassification, ViTFeatureExtractor
)
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from models.modeliing_bert import BertForSequenceClassification
from models.modeling_roberta import RobertaForSequenceClassification
from torch import nn
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import utils as utils
from datasets import load_dataset
from plot_utils.sample_points import get_robust_ind
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
# from modeling_utils import PreTrainedModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
accelerator = Accelerator()
device = accelerator.device

writer = SummaryWriter(f'runs/{time.strftime("%c")}')

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('-t', default=None, type=str)
    parser.add_argument("--dataset_name", default='cifar10', type=str)
    parser.add_argument("--task_name", default="None", type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('saved_models'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use.')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    # parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir',type=str,default='/root/tmp_dir')
    # others
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--attack_every_epoch', default=False)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time',action="store_true")

    # few-shot setting
    parser.add_argument('--random_select', default=False, type=bool, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=5, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')
    parser.add_argument('--use_fgsm', default=0, type=float, help='')
    parser.add_argument('--separate_loss', default=True, type=bool, help='log loss based on sample robustness')
    parser.add_argument('--set_percentage', default=None, type=float, help='what percentage of data should be used (based on robustness -- table 1)')
    parser.add_argument('--group', default=None, help='robustness group to be used for training')
    parser.add_argument('--do_robustness', action='store_true', help='do robustness testing')

    # robust data statistics
    parser.add_argument('--statistic_interval', default=0.35, type=float, help='')
    parser.add_argument('--dataset_len', default=None, type=int, help='')
    parser.add_argument('--with_untrained_model', default=1, type=int, help='')
    parser.add_argument('--use_cur_preds', default=0, type=int, help='whether use cur predictions or golden labels to calculate loss')
    parser.add_argument('--do_train_shuffle', default=1, type=int, help='')

    args = parser.parse_args()

    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def robust_statistics(model,train_dev_loader,train_set_len,device,use_cur_preds=True):
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["golden_label"] = cur_label.item()
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            statistics[data_index]["original_logit"] = cur_logits[cur_label.item()].item()
            statistics[data_index]["original_probability"] = nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item()

            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if args.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            if args.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif args.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                  args.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)
        total_loss = 0.0
        for astep in range(args.adv_steps):
            # 0. forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**batch, return_dict=False)[0]
            _, preds = logits.max(dim=-1)
            # 1.
            if use_cur_preds:
                losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
            else:
                losses = F.cross_entropy(logits, labels.squeeze(-1))
            # losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            loss = loss / args.adv_steps
            total_loss += loss.item()
            loss.backward()

            if astep == args.adv_steps - 1:

                for i in range(len(labels)):
                    cur_logits = logits[i]
                    cur_label = labels[i]
                    cur_pred = preds[i]
                    if use_cur_preds:
                        cur_batch_pred = cur_batch_preds[i]
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
                    else:
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
                    cur_loss = torch.mean(cur_losses)
                    statistics[data_index]["after_perturb_loss"] = cur_loss.item()
                    statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())

                    statistics[data_index]["after_perturb_logit"] = cur_logits[cur_label.item()].item()
                    statistics[data_index]["after_perturb_probability"] = nn.Softmax(dim=-1)(cur_logits)[
                        cur_label.item()].item()

                    statistics[data_index]["logit_diff"] = statistics[data_index]["after_perturb_logit"] - statistics[data_index]["original_logit"]
                    statistics[data_index]["probability_diff"] = statistics[data_index]["after_perturb_probability"] - statistics[data_index]["original_probability"]

                    statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index]["original_loss"]
                    statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"] / delta.norm(p=2,dim=(1,2),keepdim=False)[i].item()
                    data_index += 1
                break


            # 2. get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # 3. update and clip
            if args.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + args.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
                if args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                        1)
                    delta = (delta * reweights).detach()
            elif args.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                         1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding
        pbar.set_description("Doing perturbation statistics")

    return statistics


def robust_statistics_fgsm(model,train_dev_loader,train_set_len,device,use_cur_preds=True):

    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0

        # 0. forward
        embedding_init.requires_grad_()
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()
        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()
        # 3. update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                    1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                     1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()


        # 4. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        # if use_cur_preds:
        #     losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        # else:
        #     losses = F.cross_entropy(logits, labels.squeeze(-1))
        # # losses = F.cross_entropy(logits, labels.squeeze(-1))
        # loss = torch.mean(losses)
        # loss = loss / args.adv_steps
        # total_loss += loss.item()
        # loss.backward()

        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_batch_pred = cur_batch_preds[i]
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["after_perturb_loss"] = cur_loss.item()
            statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())
            statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index][
                "original_loss"]
            statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"] / \
                                                         delta.norm(p=2, dim=(1, 2), keepdim=False)[i].item()
            statistics[data_index]["delta_grad"] = delta_grad.norm(p=2, dim=(1, 2), keepdim=False)[i].item()
            data_index += 1

        pbar.set_description("Doing perturbation statistics")


    return statistics

def robust_statistics_grad(model,train_dev_loader,train_set_len,device,use_cur_preds=True):
    """
    Collect statistics
    :param model:
    :param train_dev_loader:
    :param train_set_len:
    :param device:
    :param use_cur_preds:
    :return:
    """
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0

        # 0. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()
        # 2. get gradient on delta


    model.zero_grad()

    return statistics


def collate_fn(batch):
    #to_tensor = transforms.PILToTensor()
    #col_dict = {
    #    'pixel_values': torch.stack([to_tensor(x['img']) for x in batch]),
    #    'labels': torch.tensor([x['label'] for x in batch])
    #}
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    #output_dict = {"pixel_values": pixel_values, "labels": labels}
    output_dict = {"pixel_values": pixel_values}

    #print(output_dict)
    return output_dict, labels
    #return col_dict


def preprocess_dev(example_batch):
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233, 0.24348505, 0.26158768))
    dev_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
            # normalize,
        ]
    )
    example_batch["pixel_values"] = [dev_transforms(image.convert("RGB")) for image in example_batch["img"]]
    return example_batch



def preprocess_train(example_batch):
    """Apply _train_transforms across a batch."""
    #image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    #size = (image_processor.size["height"], image_processor.size["width"])
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233, 0.24348505, 0.26158768))
    train_transforms = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]
    )

    # example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
    example_batch["pixel_values"] = [train_transforms(image) for image in example_batch["img"]]

#print(f'type of example batch {type(example_batch["pixel_values"][0])}')
    return example_batch


def process_example(example):
    inputs = feature_extractor(example['img'], return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs

#if args.dataset == 'cifar10':
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    print(f'inputtt {type(inputs)}')
    return inputs


def calc_loss(loader, epoch, model, name, args):
    vis = False
    pbar = tqdm(loader)
    model.train()
    func = globals()[args.t]
    t_min, t_max = get_t_range(args.t)
    t_max = t_max / 2
    intensity = t_min + epoch * (t_max - t_min) / 10
    experiment_name = f'{args.dataset_name}_epochs_{args.epochs}_lr_{args.lr}_bsz_{args.bsz}_t_{args.t}_dynamic'

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        cur_batch_logits = model(**model_inputs).logits
        _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        input_ids = model_inputs['pixel_values']

        if epoch == 0 and not vis:
            vis = True
            writer.add_images(name, input_ids)

        transformed_batch = torch.empty_like(input_ids)
        for i in range(input_ids.shape[0]):
            transformed = func((255*input_ids[i].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8), intensity)
            transformed_batch[i] = torch.from_numpy(transformed).permute(2, 0, 1)
        batch = {'pixel_values': transformed_batch}

        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        # if use_cur_preds:
        losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        # else:
        #     losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        pbar.set_description("Doing perturbation statistics")
    writer.add_scalar(f'{experiment_name}/AdvLoss/{name}', loss, epoch)


def separate_loss(train_dataset, epoch, model, args):
    experiment_name = f'{args.dataset_name}_epochs_{args.epochs}_lr_{args.lr}_bsz_{args.bsz}_t_{args.t}_dynamic'
    robust_ind, swing_ind, non_robust_ind = get_robust_ind(f"{experiment_name}.npy", 0.1)
    robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, robust_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
    swing_loader = DataLoader(torch.utils.data.Subset(train_dataset, swing_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
    non_robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, non_robust_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
    calc_loss(robust_loader, epoch, model, "robust", args)
    calc_loss(swing_loader, epoch, model, "swing", args)
    calc_loss(non_robust_loader, epoch, model, "non_robust", args)


def robust_statistics_perturbation(epoch, model, train_dev_loader, train_set_len, device, use_cur_preds=True):
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics = {}
    func = globals()[args.t]
    t_min, t_max = get_t_range(args.t)
    intensity = t_min + epoch * (t_max - t_min) / 10
    # print(f'intensity {intensity}')
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["golden_label"] = cur_label.item()
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            statistics[data_index]["original_logit"] = cur_logits[cur_label.item()].item()
            statistics[data_index]["original_probability"] = nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item()

            data_index+=1
        #     break
        # break
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        # word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['pixel_values']
        transformed_batch = torch.empty_like(input_ids)
        for i in range(input_ids.shape[0]):
            transformed = func((255*input_ids[i].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8), intensity)
            if i == 0:
                # print((256*np.array(input_ids[i].permute(1, 2, 0).detach().cpu())).astype(np.uint8))
                # print(f'******{transformed}******')
                writer.add_image(f"original_{args.t}_{intensity}", input_ids[i])
                writer.add_image(f"transformed_{args.t}_{intensity}", transforms.ToTensor()(transformed))
            transformed_batch[i] = torch.from_numpy(transformed).permute(2, 0, 1)
            #transformed = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0)
            # print(transformed.shape)
            #transformed_batch.append(transformed)
            # print(transformed.shape)
        #print(transformed_batch)
        #temp = np.array(transformed_batch)
        #print(temp.shape)
        #transformed_batch = torch.vstack(torch.from_numpy(np.array(transformed_batch)))
        # print(transformed_batch.shape)
        batch = {'pixel_values': transformed_batch}

        # total_loss = 0.0


        # for astep in range(args.adv_steps):
        #     # 0. forward
        #     delta.requires_grad_()
        #     batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        # total_loss += loss.item()
        # loss.backward()
        #
        #     if astep == args.adv_steps - 1:

        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_batch_pred = cur_batch_preds[i]
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["after_perturb_loss"] = cur_loss.item()
            statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())

            statistics[data_index]["after_perturb_logit"] = cur_logits[cur_label.item()].item()
            statistics[data_index]["after_perturb_probability"] = nn.Softmax(dim=-1)(cur_logits)[
                cur_label.item()].item()

            statistics[data_index]["logit_diff"] = statistics[data_index]["after_perturb_logit"] - statistics[data_index]["original_logit"]
            statistics[data_index]["probability_diff"] = statistics[data_index]["after_perturb_probability"] - statistics[data_index]["original_probability"]

            statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index]["original_loss"]
            statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"]
            #/ delta.norm(p=2,dim=(1,2),keepdim=False)[i].item()
            data_index += 1
        #     break
        # break

        pbar.set_description("Doing perturbation statistics")
    return statistics


def finetune(args):
        # gpu, ngpus_per_node, args):
    set_seed(args.seed)
    experiment_name = f'{args.dataset_name}_epochs_{args.epochs}_lr_{args.lr}_bsz_{args.bsz}_t_{args.t}_dynamic'
    # args.gpu = gpu
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))
    #
    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    #

    # output_dir = Path(args.output_dir)
    # if not output_dir.exists():
    #     logger.info(f'Making checkpoint directory: {output_dir}')
    #     output_dir.mkdir(parents=True)
    # elif args.not_force_overwrite:
    #     return
    # log_file = os.path.join(output_dir, 'INFO.log')

    if args.dataset_name == "imdb":
        num_labels = 2
        args.num_labels = 2
        output_mode = "classification"
    elif args.dataset_name == "ag_news":
        num_labels = 4
        args.num_labels = 4
        output_mode = "classification"
    elif args.dataset_name == "cifar10":
        num_labels = 10
        args.num_labels = 10
        output_mode = 'classification'

    # pre-trained config/tokenizer/model
    device = accelerator.device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels,mirror='tuna')

    if args.model_name == 'google/vit-base-patch16-224':
        tokenizer = AutoFeatureExtractor.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name =="bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)
    #elif args.model_name == "google/vit-base-patch16-224":
    #    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        #hidden_size = model.config.hidden_size

        ## Define the new output dimension
        #new_output_dim = 10  # Replace 10 with your desired output dimension

        ## Replace the last layer with a new linear layer
        #new_last_layer = torch.nn.Linear(hidden_size, new_output_dim)
        #model.set_classifier(new_last_layer)
    elif args.model_name == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        if args.model_name != 'google/vit-base-patch16-224':
            model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)

    # prepare datasets
    # logger.info(utils.say())
    if args.dataset_name == 'cifar10':
        collator = collate_fn
    else:
        collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name=="SetFit/20_newsgroups":
        args.task_name = None
        args.valid = "test"

    elif args.task_name=="mnli":
        args.valid = "validation_matched"
        output_mode = "classification"

    if args.dataset_name == 'cifar10':
        #train_dataset = utils.VisionDataset(args, name_or_dataset=args.dataset_name, split="train")
        #train_dataset = CIFAR10('data', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        train_dataset = load_dataset('cifar10', split='train')
        #print(train_dataset[0])
        train_dataset = train_dataset.with_transform(preprocess_train)
        dev_dataset = load_dataset('cifar10', split='test')
        dev_dataset = dev_dataset.with_transform(preprocess_dev)
        #
        # if args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #     dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False, drop_last=True)
        # else:
        train_sampler = None
        dev_sampler = None


        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=args.do_train_shuffle
                , collate_fn=collator, sampler=train_sampler
                )
        train_dev_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator, sampler=dev_sampler)


        if args.set_percentage:
            robust_ind, swing_ind, non_robust_ind = get_robust_ind(f"{experiment_name}.npy", args.set_percentage)
            robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, robust_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
            swing_loader = DataLoader(torch.utils.data.Subset(train_dataset, swing_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
            non_robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, non_robust_ind), batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
            if args.group == 'robust':
                train_loader = train_dev_loader = robust_loader
            elif args.group == 'swing':
                train_loader = train_dev_loader = swing_loader
            elif args.group == 'non_robust':
                train_loader = train_dev_loader = non_robust_loader

        if args.do_test:
            #test_dataset = utils.VisionDataset(args, name_or_dataset=args.dataset_name, split='test')
            #test_dataset = CIFAR10(data, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
            test_dataset = load_dataset('cifar10', split='test')
            test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator, sampler=dev_sampler)

        logger.info("train dataset length: "+ str(len(train_dataset)))
        logger.info("dev dataset length: "+ str(len(dev_dataset)))
        logger.info("train loader length (approx): "+ str(len(train_loader) * args.bsz))

        train_dev_dict = {}
        labels = train_dataset.features["label"].names
        label2id = {label: str(i) for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(args.model_name, num_labels=len(labels), i2label=id2label, label2id=label2id, finetuning_task="image-classification")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        #
        #
        # if args.distributed:
        #     # For multiprocessing distributed, DistributedDataParallel constructor
        #     # should always set the single device scope, otherwise,
        #     # DistributedDataParallel will use all available devices.
        #     if torch.cuda.is_available():
        #         if args.gpu is not None:
        #             torch.cuda.set_device(args.gpu)
        #             model.cuda(args.gpu)
        #             # When using a single GPU per process and per
        #             # DistributedDataParallel, we need to divide the batch size
        #             # ourselves based on the total number of GPUs of the current node.
        #             args.batch_size = int(args.batch_size / ngpus_per_node)
        #             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        #         else:
        #             model.cuda()
        #             # DistributedDataParallel will divide and allocate batch_size to all
        #             # available GPUs if device_ids are not set
        #             model = torch.nn.parallel.DistributedDataParallel(model)
        # elif args.gpu is not None and torch.cuda.is_available():
        #     torch.cuda.set_device(args.gpu)
        #     model = model.cuda(args.gpu)
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     model = model.to(device)
        # else:
        #     # DataParallel will divide and allocate batch_size to all available GPUs
        #     if args.model_name.startswith('alexnet') or args.model_name.startswith('vgg'):
        #         model.features = torch.nn.DataParallel(model.features)
        #         model.cuda()
        #     else:
        #         model = torch.nn.DataParallel(model).cuda()
        #
        # if torch.cuda.is_available():
        #     if args.gpu:
        #         device = torch.device('cuda:{}'.format(args.gpu))
        #     else:
        #         device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # else:
        #     device = torch.device("cpu")


        model.to(device)
        print(model)
        #dev_dataset = utils.VisionDataset(args, name_or_dataset=args.dataset_name, split="test")
        #dev_dataset = CIFAR10('data', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)


    else:

        train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
        if args.dataset_len is not None and args.dataset_len < len(train_dataset):
            # total_size =
            train_dataset.dataset = train_dataset.dataset.train_test_split(1,args.dataset_len,seed=42)["train"]
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=args.do_train_shuffle, collate_fn=collator)
    
        train_dev_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    
        train_dev_dict = {}
        logger.info("train dataset length: "+ str(len(train_dataset)))
        # for dev
        dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                subset=args.task_name, split=args.valid)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        
        logger.info("dev dataset length: "+ str(len(dev_dataset)))
    
        # for test
        if args.do_test:
            test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
            test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=args.adam_epsilon,
                      correct_bias=args.bias_correction
    )


    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    epoch_steps = len(train_dataset) // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    robust_statistics_dict = {}

    # num_record_steps = num_training_steps//args.statistic_interval + 1
    # robust_statistics_records = [[] for _ in range(num_record_steps)]

    try:
        # import time
        best_accuracy = 0
        global_step = 0
        for epoch in range(args.epochs):
            # if args.distributed:
            #     train_sampler.set_epoch(epoch)
            avg_loss = utils.ExponentialMovingAverage()
            model.train()
            pbar = tqdm(train_loader)
            for idx, (model_inputs, labels) in enumerate(train_loader):
                #print(f"model input {model_inputs}, labels {labels}")
                if args.with_untrained_model:
                    # print(f"global step {global_step} stat interval {args.statistic_interval} epoch step {epoch_steps} "
                    #       f"true? {global_step % int(args.statistic_interval * epoch_steps) == 0}")
                    if global_step % int(args.statistic_interval * epoch_steps) == 0:
                        if args.separate_loss:
                            separate_loss(train_dataset, epoch, model, args)
                            # and global_step!=0\

                        else:
                            if args.use_fgsm:
                                cur_robust_statistics = robust_statistics_fgsm(model, train_dev_loader,
                                                                      train_set_len=len(train_dataset), device=device,
                                                                      use_cur_preds=args.use_cur_preds)
                            else:
                                cur_robust_statistics = robust_statistics_perturbation(epoch, model,train_dev_loader,train_set_len=len(train_dataset),device=device,use_cur_preds=args.use_cur_preds)
                                #robust_statistics(model,train_dev_loader,train_set_len=len(train_dataset),device=device,use_cur_preds=args.use_cur_preds)

                            robust_statistics_dict[global_step] = cur_robust_statistics
                        # print('robust stats updated')
                        # print(robust_statistics_dict.keys())
                else:
                    if global_step % int(args.statistic_interval * epoch_steps) == 0 and global_step!=0:
                        if args.use_fgsm:
                            cur_robust_statistics = robust_statistics_fgsm(model, train_dev_loader,
                                                                           train_set_len=len(train_dataset),
                                                                           device=device,
                                                                           use_cur_preds=args.use_cur_preds)
                        else:
                            cur_robust_statistics = robust_statistics_perturbation(epoch, model,train_dev_loader,train_set_len=len(train_dataset),device=device,use_cur_preds=args.use_cur_preds)
                                #robust_statistics(model, train_dev_loader, train_set_len=len(train_dataset), device=device, use_cur_preds=args.use_cur_preds)
                        robust_statistics_dict[global_step] = cur_robust_statistics
                        print('robust stats updated')

                batch_loss=0
                # model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                model_inputs = model_inputs['pixel_values']
                # labels = labels.to(device)
                model.zero_grad()
                logits = model(model_inputs, return_dict=False)[0]
                _, preds = logits.max(dim=-1)

                losses = F.cross_entropy(logits,labels.squeeze(-1))
                loss = torch.mean(losses)

                # loss2  = model(**model_inputs,return_dict=False)
                batch_loss=loss.item()
                # loss.backward()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                avg_loss.update(batch_loss)

                global_step+=1
            pbar.set_description(f'epoch: {epoch: d}, '
                                 f'Transformation: {args.t}, '
                                 f'loss: {avg_loss.get_metric(): 0.4f}, '
                                 f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
            writer.add_scalar(f'{experiment_name}/Train/Loss', loss, epoch)

            # s = Path(str(output_dir) + '/epoch' + str(epoch))
            # if not s.exists():
            #     s.mkdir(parents=True)
            # model.save_pretrained(s)
            # tokenizer.save_pretrained(s)
            # torch.save(args, os.path.join(s, "training_args.bin"))

            if args.do_eval and not args.cal_time:
                logger.info('Evaluating...')
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        model_inputs = model_inputs['pixel_values']
                        labels = labels.to(device)
                        logits = model(model_inputs,return_dict=False)[0]
                        _, preds = logits.max(dim=-1)

                        losses = F.cross_entropy(logits,labels.squeeze(-1))
                        loss = torch.mean(losses)
                        batch_loss=loss.item()
                        avg_loss.update(batch_loss)

                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = 100 * correct / (total + 1e-13)
                logger.info(f'Epoch: {epoch}, '
                            f'Transformation: {args.t}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')

                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    # model.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch

            logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')
            writer.add_scalar(f'{experiment_name}/Test/Loss', loss, epoch)

            if args.do_robustness and not args.cal_time:
                logger.info('Calculating robustness...')
                model.eval()
                correct = 0
                total = 0
                func = globals()[args.t]
                t_min, t_max = get_t_range(args.t)
                intensity = t_min + epoch * (t_max - t_min) / 10
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        model_inputs = model_inputs['pixel_values']
                        labels = labels.to(device)

                        transformed_batch = torch.empty_like(model_inputs)
                        for i in range(model_inputs.shape[0]):
                            transformed = func((255*model_inputs[i].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8), intensity)
                            # if i == 0:
                                # print((256*np.array(input_ids[i].permute(1, 2, 0).detach().cpu())).astype(np.uint8))
                                # print(f'******{transformed}******')
                                # writer.add_image(f"original_{args.t}_{intensity}", model_inputs[i])
                                # writer.add_image(f"transformed_{args.t}_{intensity}", transforms.ToTensor()(transformed))
                            transformed_batch[i] = torch.from_numpy(transformed).permute(2, 0, 1)
                        batch = {'pixel_values': transformed_batch}

                        logits = model(**batch, return_dict=False)[0]
                        _, preds = logits.max(dim=-1)
                        losses = F.cross_entropy(logits, labels.squeeze(-1))
                        loss = torch.mean(losses)
                        batch_loss=loss.item()
                        avg_loss.update(batch_loss)

                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = 100 * correct / (total + 1e-13)
                logger.info(f'Epoch: {epoch}, '
                            f'Transformation: {args.t}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Robustness: {accuracy}')
                #
                if accuracy > best_accuracy:
                    logger.info('Best robustness so far.')
                    # model.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch

            logger.info(f'Best robustness: {best_accuracy} in Epoch: {best_dev_epoch}')
            # writer.add_scalar(f'{experiment_name}/Test/Loss', loss, epoch)

        # save statistics


        if args.output_dir=='/root/tmp_dir':
            if args.use_fgsm:
                np.save(
                    'fgsm/robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
                    .format(args.model_name, args.dataset_name, args.task_name, args.seed, args.do_train_shuffle,
                            args.dataset_len,
                            args.adv_steps, args.adv_lr, args.epochs, args.lr,
                            args.statistic_interval, args.with_untrained_model, args.use_cur_preds
                            ),
                    robust_statistics_dict)
            else:
                if not args.separate_loss:
                    np.save(f'{experiment_name}.npy',
                    #'robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
                        # .format(args.model_name.split('/')[-1],args.dataset_name,args.task_name,args.seed,args.do_train_shuffle,
                        #         args.dataset_len,
                        #         args.adv_steps,args.adv_lr,args.epochs,args.lr,
                        #         args.statistic_interval,args.with_untrained_model,args.use_cur_preds
                                #),
                        robust_statistics_dict)
        else:
            if not args.separate_loss:
                np.save(f'{experiment_name}.npy', robust_statistics_dict)
            # with open('test_dict.txt', 'w') as f:
            #     f.write(json.dumps(robust_statistics_dict))

    except KeyboardInterrupt:
        logger.info('Interrupted...')


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #
    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    #
    # if torch.cuda.is_available():
    #     ngpus_per_node = torch.cuda.device_count()
    # else:
    #     ngpus_per_node = 1
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(finetune, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     finetune(args.gpu, ngpus_per_node, args)


    finetune(args)
