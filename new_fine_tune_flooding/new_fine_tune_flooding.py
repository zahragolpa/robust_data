# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : new_fine_tune_flooding.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 19:03
@Author   : Zhiheng Xi
"""

import argparse
import logging
import os
from pathlib import Path
import random

import datasets.arrow_dataset
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer, AutoFeatureExtractor
)

from models.modeling_roberta import RobertaForSequenceClassification
from models.modeliing_bert import BertForSequenceClassification
from transformations import get_t_range
from datasets import load_dataset
from transformers import (
    AdamW, AutoConfig, AutoTokenizer, AutoFeatureExtractor, ViTForImageClassification, ViTFeatureExtractor
)
from accelerate import Accelerator
import torchvision.transforms as transforms
from datasets import arrow_dataset
from torch.utils.tensorboard import SummaryWriter
accelerator = Accelerator()
device = accelerator.device
import utils as utils
import time
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer  # modified

# from modeling_utils import PreTrainedModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import statistic_utils as statistic_utils
from finetune_with_select_data import save_model_one_step, save_model_one_epoch, generate_data_indices, \
    do_textattack_attack, do_pgd_attack, evaluate, select_data, sort_df_by_metric, do_perturb

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
writer = SummaryWriter(f'runs/{time.strftime("%c")}')

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument("--dataset_name", default='cifar10', type=str)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models'))
    parser.add_argument('--num_labels', type=int, default=10)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--eval_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    # parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./saved_models')

    parser.add_argument('--reinit_classifier', default=0, type=int, help="")
    parser.add_argument('--freeze_bert', default=0, type=int, help="")

    # others
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--attack_every_epoch', default=0, type=int,
                        help="10:攻击train dataset1；20：攻击 train dataset2；否则攻击dev dataset")
    parser.add_argument('--preturb_every_epoch', default=True, type=bool,
                        help="measure robustness every epoch")
    parser.add_argument('--attack_every_step', default=0, type=int)
    parser.add_argument('-f', '--not_force_overwrite', action="store_true")  # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time', action="store_true")

    # few-shot setting
    parser.add_argument('--random_select', default=0, type=int, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')

    # select data
    parser.add_argument('--statistics_source', type=str, default=None)
    parser.add_argument('--select_metric', type=str, default="top_right")
    parser.add_argument('--select_metric2', type=str, default="loss_diff_mean_r")
    parser.add_argument('--select_ratio', type=float, default=0.1)
    parser.add_argument('--select_ratio2', type=float, default=0.1)
    parser.add_argument('--select_from_two_end', type=int, default=0)
    parser.add_argument('--ratio_proportion', type=float, default=0.5)
    parser.add_argument('--do_balance_labels', type=int, default=1)
    parser.add_argument('--with_untrained_model', default=1, type=int, help='')
    parser.add_argument('--use_cur_preds', default=0, type=int,
                        help='whether use cur predictions or golden labels to calculate loss')
    parser.add_argument('--only_original_pred', default=1, type=int, help='')
    parser.add_argument('--cycle_train', default=0, type=int, help='train for continuous epochs if cycle_train>0')
    parser.add_argument('--save_steps', default=-1, type=float, help="")
    parser.add_argument('--show_data', default=-1, type=int, help="")

    # attack
    parser.add_argument('--do_attack', action="store_true")
    parser.add_argument('--attack_all', action="store_true")
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.15, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)
    parser.add_argument("--results_file", default='attack_log.csv', type=str)
    parser.add_argument("--num_examples", default=1000, type=int)
    parser.add_argument("--attack_method", default="textfooler", type=str)
    # pgd attack
    parser.add_argument("--do_pgd_attack", default=0, type=int)
    parser.add_argument("--do_perturb", default=1, type=int)
    parser.add_argument("--exp", default=None, type=str)

    parser.add_argument('--pgd_step', type=int, default=5)
    parser.add_argument('--pgd_lr', type=float, default=0.05)
    # freelb
    parser.add_argument('--pgd_adv_steps', type=int, default=5)
    parser.add_argument('--pgd_adv_steps2', type=int, default=10)
    parser.add_argument('--pgd_adv_lr', type=float, default=0.03)
    parser.add_argument('--pgd_adv_lr2', type=float, default=0.03)
    parser.add_argument('--do_pgd_training', type=int, default=0)

    # parser.add_argument('--adv_steps', default=5, type=int,
    #                     help='Number of gradient ascent steps for the adversary')
    # parser.add_argument('--adv_lr', default=0.03, type=float,
    #                     help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    # new finetune
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--soft_label', type=float, default=1)
    parser.add_argument('--b', type=float, default=0.2)

    args = parser.parse_args()
    args.statistics_source = f"../{args.exp}.npy"
    # if args.balance_labels:
    #     print("s")
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '../..'
    print(args.__dict__)

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


def indexed_collate(batch):
    # to_tensor = transforms.PILToTensor()
    # col_dict = {
    #    'pixel_values': torch.stack([to_tensor(x['img']) for x in batch]),
    #    'labels': torch.tensor([x['label'] for x in batch])
    # }
    # print(f'batch 0 : {batch[0]}')
    # print(batch)
    pixel_values = torch.stack([example["pixel_values"] for example in batch[0]])
    labels = [torch.tensor([example["label"] for example in batch[0]]), batch[1]]
    # output_dict = {"pixel_values": pixel_values, "labels": labels}
    output_dict = {"pixel_values": pixel_values}
    # print(output_dict)
    return output_dict, labels


def collate_fn(batch):
    #to_tensor = transforms.PILToTensor()
    #col_dict = {
    #    'pixel_values': torch.stack([to_tensor(x['img']) for x in batch]),
    #    'labels': torch.tensor([x['label'] for x in batch])
    #}
    # print(batch)
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    #output_dict = {"pixel_values": pixel_values, "labels": labels}
    output_dict = {"pixel_values": pixel_values}

    #print(output_dict)
    return output_dict, labels
    #return col_dict


def preprocess_dev(example_batch):
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                                     std=(0.24703233, 0.24348505, 0.26158768))
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
    # image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    # size = (image_processor.size["height"], image_processor.size["width"])
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                                     std=(0.24703233, 0.24348505, 0.26158768))
    train_transforms = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]
    )

    # example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
    example_batch["pixel_values"] = [train_transforms(image) for image in example_batch["img"]]
    # print(f'example_batch {example_batch}')
    # print(f'type of example batch {type(example_batch["pixel_values"][0])}')
    return example_batch


data_indices_to_new_id = {}
new_id_to_data_indices = {}


def finetune(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif args.not_force_overwrite:
        return
    log_file = os.path.join(output_dir, 'INFO.log')

    if args.dataset_name == "imdb":
        num_labels = 2
        output_mode = "classification"
    elif args.dataset_name == "ag_news":
        num_labels = 4
        output_mode = "classification"

    elif args.dataset_name == 'cifar10':
        num_labels = 10

    # pre-trained config/tokenizer/model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels, mirror='tuna')
    if args.model_name == 'google/vit-base-patch16-224-in21k':
        tokenizer = AutoFeatureExtractor.from_pretrained(args.model_name)
    else:
        tokenizer = AutoFeatureExtractor.from_pretrained(args.model_name)
    if args.model_name == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
    elif args.model_name == "google/vit-base-patch16-224-in21k":
        train_dataset = load_dataset('cifar10', split='train')
        # print(train_dataset[0])
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

        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True
                                  , collate_fn=collate_fn, sampler=train_sampler
                                  )
        train_dev_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collate_fn,
                                sampler=dev_sampler)

        # if args.set_percentage:
        #     robust_ind, swing_ind, non_robust_ind = get_robust_ind(f"{experiment_name}.npy", args.set_percentage)
        #     robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, robust_ind), batch_size=args.bsz,
        #                                shuffle=False, collate_fn=collate_fn)
        #     swing_loader = DataLoader(torch.utils.data.Subset(train_dataset, swing_ind), batch_size=args.bsz,
        #                               shuffle=False, collate_fn=collate_fn)
        #     non_robust_loader = DataLoader(torch.utils.data.Subset(train_dataset, non_robust_ind), batch_size=args.bsz,
        #                                    shuffle=False, collate_fn=collate_fn)
        #     if args.group == 'robust':
        #         train_loader = train_dev_loader = robust_loader
        #     elif args.group == 'swing':
        #         train_loader = train_dev_loader = swing_loader
        #     elif args.group == 'non_robust':
        #         train_loader = train_dev_loader = non_robust_loader

        if args.do_test:
            # test_dataset = utils.VisionDataset(args, name_or_dataset=args.dataset_name, split='test')
            # test_dataset = CIFAR10(data, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
            test_dataset = load_dataset('cifar10', split='test')
            test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collate_fn,
                                     sampler=dev_sampler)

        logger.info("train dataset length: " + str(len(train_dataset)))
        logger.info("dev dataset length: " + str(len(dev_dataset)))
        logger.info("train loader length (approx): " + str(len(train_loader) * args.bsz))

        train_dev_dict = {}
        labels = train_dataset.features["label"].names
        label2id = {label: str(i) for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(args.model_name, num_labels=len(labels), i2label=id2label,
                                            label2id=label2id, finetuning_task="image-classification")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)

    elif args.model_name == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)

    model.to(device)
    if args.reinit_classifier:
        model.reinit_classifier()
    if args.freeze_bert:
        model.freeze_Bert()

    # prepare datasets
    # logger.info(utils.say())
    # collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    collator = collate_fn
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name == "SetFit/20_newsgroups":
        args.task_name = None
        args.valid = "test"

    # train_dataset = utils.Huggingface_dataset_with_data_id(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    assert args.statistics_source != None
    result_data_indices, selected_label_nums = generate_data_indices(args, train_dataset, args.select_metric,
                                                                     args.select_ratio)
    # print(str(result_data_indices))
    if args.dataset_name == "imdb":
        train_dataset.updata_idx(data_indices_to_new_id, new_id_to_data_indices)

    if args.show_data > 0:
        import csv
        print("Metric:{}".format(args.select_metric))
        for i in range(args.show_data):
            print("sentence: {}, label: {}".format(train_dataset.dataset["sentence"][i],
                                                   train_dataset.dataset["label"][i]))
            print()
            print()
            show_data_dir = "/root/Robust_Data/analysis_experiments/show_data/"
            show_data_file = "/root/Robust_Data/analysis_experiments/show_data/show_data.csv"
            show_data_format = [
                "select_metric", "seed", "sentence", "label", "order_in_cur_metric"
            ]
            if not os.path.exists(show_data_file):
                # os.makedirs(show_data_dir)
                out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                csv_writer = csv.writer(out_csv)
                cur_row = [i for i in show_data_format]
                csv_writer.writerow(cur_row)
                out_csv.close()

            out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
            csv_writer = csv.writer(out_csv)
            cur_row = []
            cur_row.append(args.select_metric)
            cur_row.append(args.seed)
            cur_row.append(train_dataset.dataset["sentence"][i])
            cur_row.append(train_dataset.dataset["label"][i])
            cur_row.append(i)
            csv_writer.writerow(cur_row)
            out_csv.close()

        return

    print(str(selected_label_nums))
    args.selected_label_nums = str(selected_label_nums)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    logger.info("train dataset length: " + str(len(train_dataset)))

    if args.model_name != "google/vit-base-patch16-224-in21k":
        # for dev
        dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                subset=args.task_name, split=args.valid)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

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
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    log_format = ["statistics_source",
                  "select_metric",
                  "select_ratio", "ratio_proportion",
                  "selected_label_nums",
                  "lr",
                  "seed", "epochs",
                  "pgd_step", "pgd_lr",
                  "clean", "pgd_aua", "attack_method", "beta", "b"
                  ]

    one_epoch_steps = int(len(train_dataset) // args.bsz)
    if args.save_steps < 1 and args.save_steps > 0:
        args.save_steps = int(one_epoch_steps * args.save_steps)
    save_steps = args.save_steps
    try:
        best_accuracy = 0
        global_step = 0
        for epoch in range(args.epochs):
            avg_loss = utils.ExponentialMovingAverage()

            model.train()
            # train 1 epoch
            # train with flooding
            global_step = train_one_epoch_new(args, avg_loss, device, epoch, model, optimizer, scheduler, train_dataset,
                                              global_step, save_steps, tokenizer, result_data_indices, beta=args.beta,
                                              b=args.b)

            # save model
            # save_model_one_epoch(args, epoch, model, output_dir, tokenizer)
            # eval model
            if args.do_eval and not args.cal_time:
                # accuracy, clean_loss = evaluate(dev_loader, device, model)
                # logger.info(f'Epoch: {epoch}, '
                #             f'Loss: {avg_loss.get_metric(): 0.4f}, '
                #             f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                #             f'Accuracy: {accuracy}')
                # if accuracy > best_accuracy:
                #     logger.info('Best performance so far.')
                #     # model.save_pretrained(output_dir)
                #     # tokenizer.save_pretrained(output_dir)
                #     # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     best_accuracy = accuracy
                #     best_dev_epoch = epoch
                # writer.add_scalar(f'{args.exp}/Test/Loss', clean_loss.get_metric(), epoch)
                logger.info('Evaluating...')
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        model_inputs = model_inputs['pixel_values']
                        labels = labels.to(device)
                        logits = model(model_inputs, return_dict=False)[0]
                        _, preds = logits.max(dim=-1)

                        losses = F.cross_entropy(logits, labels.squeeze(-1))
                        loss = torch.mean(losses)
                        batch_loss = loss.item()
                        avg_loss.update(batch_loss)

                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = 100 * correct / (total + 1e-13)
                logger.info(f'Clean Aua: {accuracy}')
                logger.info(f'Clean Loss: {avg_loss.get_metric()}')
                # logger.info(f'Epoch: {epoch}, '
                #             f'Transformation: {args.t}, '
                #             f'Loss: {avg_loss.get_metric(): 0.4f}, '
                #             f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                #             f'Accuracy: {accuracy}')

                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    # model.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch

                logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')
                writer.add_scalar(f'{args.exp}/Test/Loss', avg_loss.get_metric(), epoch)

            if args.preturb_every_epoch:
                pgd_aua, pgd_loss = do_perturb(dev_loader, device, model, args.exp,
                                               adv_steps=args.pgd_step,
                                               adv_lr=args.pgd_lr,
                                               adv_init_mag=args.adv_init_mag,
                                               adv_max_norm=args.adv_max_norm,
                                               adv_norm_type=args.adv_norm_type, writer=writer
                                               )
                optimizer.zero_grad()
                args.pgd_aua = pgd_aua
                args.pgd_loss = pgd_loss.get_metric()
                writer.add_scalar(f'{args.exp}/robustness', pgd_aua, epoch)
        if args.attack_every_step > 0:
            for step in range(num_training_steps):
                if step % args.save_steps == 0:

                    one_epoch_steps = int(len(train_dataset) // args.bsz)

                    epoch = int(step // one_epoch_steps)
                    logger.info("current step:{},current epoch:{}".format(step, epoch))
                    if epoch % (args.cycle_train * 2) < args.cycle_train:  #
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric, args.select_ratio))
                        cur_select_metric = args.select_metric
                        args.cur_select_metric = cur_select_metric

                    else:
                        logger.info(
                            "current metric:{},cureent ratio:{}".format(args.select_metric2, args.select_ratio2))
                        cur_select_metric = args.select_metric2
                        args.cur_select_metric = cur_select_metric
                    args.cur_epoch = epoch
                    args.cur_step = step
                    s = Path(str(output_dir) + '/step' + str(step))
                    if args.model_name == "bert-base-uncased":
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    elif args.model_name == "roberta-base":
                        model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                    else:
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    model.to(device)
                    if args.do_eval and not args.cal_time:
                        clean, clean_loss = evaluate(dev_loader, device, model)
                        args.clean = clean

                        args.clean_loss = clean_loss.get_metric()
                    else:
                        args.clean = 0
                        args.clean_loss = 0
                    if args.do_pgd_attack:
                        pgd_aua, pgd_loss = do_pgd_attack(dev_loader, device, model,
                                                          adv_steps=args.pgd_step,
                                                          adv_lr=args.pgd_lr,
                                                          adv_init_mag=args.adv_init_mag,
                                                          adv_max_norm=args.adv_max_norm,
                                                          adv_norm_type=args.adv_norm_type
                                                          )
                        optimizer.zero_grad()
                        args.pgd_aua = pgd_aua
                        args.pgd_loss = pgd_loss.get_metric()
                    elif args.do_perturb:
                        pgd_aua, pgd_loss = do_perturb(dev_loader, device, model,
                                                       adv_steps=args.pgd_step,
                                                       adv_lr=args.pgd_lr,
                                                       adv_init_mag=args.adv_init_mag,
                                                       adv_max_norm=args.adv_max_norm,
                                                       adv_norm_type=args.adv_norm_type, writer=writer
                                                       )
                        optimizer.zero_grad()
                        args.pgd_aua = pgd_aua
                        args.pgd_loss = pgd_loss.get_metric()

                    else:
                        args.pgd_aua = 0
                        args.pgd_loss = 0

                    do_textattack_attack(args, model, tokenizer,
                                         do_attack=args.do_attack,
                                         attack_seed=42,
                                         attack_all=args.attack_all,
                                         attack_method="textfooler",
                                         attack_every_epoch=False,
                                         attack_every_step=True,
                                         log_format=[i for i in log_format]

                                         )

        elif args.attack_every_epoch > 0:
            for epoch in range(args.epochs):
                logger.info("current epoch:{}".format(epoch))
                if args.cycle_train > 0:
                    if epoch % (args.cycle_train * 2) < args.cycle_train:  #
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric, args.select_ratio))
                        cur_select_metric = args.select_metric
                        args.cur_select_metric = cur_select_metric

                    else:
                        logger.info(
                            "current metric:{},cureent ratio:{}".format(args.select_metric2, args.select_ratio2))
                        cur_select_metric = args.select_metric2
                        args.cur_select_metric = cur_select_metric
                else:
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric, args.select_ratio))
                    cur_select_metric = args.select_metric
                    args.cur_select_metric = cur_select_metric

                args.cur_epoch = epoch
                s = Path(str(output_dir) + '/epoch' + str(epoch))
                if args.model_name == "bert-base-uncased":
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                elif args.model_name == "google/vit-base-patch16-224-in21k":
                    model = ViTForImageClassification.from_pretrained(s, config=config)
                elif args.model_name == "roberta-base":
                    model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                else:
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                model.to(device)
                if args.do_eval and not args.cal_time:
                    if args.attack_every_epoch == 10:
                        args.attack_dataset_metric = args.select_metric
                        loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    elif args.attack_every_epoch == 20:
                        args.attack_dataset_metric = args.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    else:
                        args.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    clean, clean_loss = evaluate(loader, device, model)
                    args.clean = clean
                    args.clean_loss = clean_loss.get_metric()
                else:
                    args.clean = 0
                    args.clean_loss = 0
                if args.do_pgd_attack:
                    if args.attack_every_epoch == 10:
                        args.attack_dataset_metric = args.select_metric
                        loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    elif args.attack_every_epoch == 20:
                        args.attack_dataset_metric = args.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    else:
                        args.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    pgd_aua, pgd_loss = do_pgd_attack(loader, device, model,
                                                      adv_steps=args.pgd_step,
                                                      adv_lr=args.pgd_lr,
                                                      adv_init_mag=args.adv_init_mag,
                                                      adv_max_norm=args.adv_max_norm,
                                                      adv_norm_type=args.adv_norm_type
                                                      )
                    optimizer.zero_grad()
                    args.pgd_aua = pgd_aua
                    args.pgd_loss = pgd_loss.get_metric()
                else:
                    args.pgd_aua = 0
                    args.pgd_loss = 0

                do_textattack_attack(args, model, tokenizer,
                                     do_attack=args.do_attack,
                                     attack_seed=42,
                                     attack_all=args.attack_all,
                                     attack_method="textfooler",
                                     attack_every_epoch=True,
                                     attack_every_step=False,
                                     log_format=[i for i in log_format]
                                     )

        if (not args.attack_every_epoch and not args.attack_every_step and args.do_attack):
            if args.do_eval and not args.cal_time:
                args.attack_dataset_metric = "test_set"
                loader = dev_loader
                clean, clean_loss = evaluate(loader, device, model)
                args.clean = clean
                args.clean_loss = clean_loss
            else:
                args.clean = 0
                args.clean_loss = 0
            if args.do_pgd_attack:
                pgd_aua, pgd_loss = do_pgd_attack(dev_loader, device, model,
                                                  adv_steps=args.pgd_step,
                                                  adv_lr=args.pgd_lr,
                                                  adv_init_mag=args.adv_init_mag,
                                                  adv_max_norm=args.adv_max_norm,
                                                  adv_norm_type=args.adv_norm_type
                                                  )
                args.pgd_aua = pgd_aua
                args.pgd_loss = pgd_loss
            elif args.do_perturb:
                pgd_aua, pgd_loss = do_perturb(dev_loader, device, model, args.exp,
                                               adv_steps=args.pgd_step,
                                               adv_lr=args.pgd_lr,
                                               adv_init_mag=args.adv_init_mag,
                                               adv_max_norm=args.adv_max_norm,
                                               adv_norm_type=args.adv_norm_type, writer=writer
                                               )
                optimizer.zero_grad()
                args.pgd_aua = pgd_aua
                args.pgd_loss = pgd_loss.get_metric()
            else:
                args.pgd_aua = 0
                args.pgd_loss = 0
            optimizer.zero_grad()
            do_textattack_attack(args, model, tokenizer,
                                 do_attack=args.do_attack,
                                 attack_seed=42,
                                 attack_all=args.attack_all,
                                 attack_method="textfooler",
                                 attack_every_epoch=args.attack_every_epoch,
                                 log_format=[i for i in log_format]
                                 )
        elif (not args.attack_every_epoch and not args.attack_every_step and not args.do_attack and args.do_pgd_attack):
            pgd_aua, pgd_loss = do_pgd_attack(dev_loader, device, model,
                                              adv_steps=args.pgd_step,
                                              adv_lr=args.pgd_lr,
                                              adv_init_mag=args.adv_init_mag,
                                              adv_max_norm=args.adv_max_norm,
                                              adv_norm_type=args.adv_norm_type,
                                              )
            optimizer.zero_grad()
            args.pgd_aua = pgd_aua
            args.pgd_loss = pgd_loss
        if args.do_perturb:
            pgd_aua, pgd_loss = do_perturb(dev_loader, device, model, args.exp,
                                           adv_steps=args.pgd_step,
                                           adv_lr=args.pgd_lr,
                                           adv_init_mag=args.adv_init_mag,
                                           adv_max_norm=args.adv_max_norm,
                                           adv_norm_type=args.adv_norm_type, writer=writer
                                           )
            optimizer.zero_grad()
            args.pgd_aua = pgd_aua
            args.pgd_loss = pgd_loss.get_metric()

    except KeyboardInterrupt:
        logger.info('Interrupted...')


def pgd_one_epoch_new(args, avg_loss, device, epoch, model, optimizer, scheduler,
                      train_loader, global_step, save_steps, tokenizer,
                      adv_init_mag, adv_norm_type, adv_steps, adv_lr,
                      adv_max_norm, result_data_indices, adv_steps2=None, adv_lr2=None, beta=1
                      ):
    pbar = tqdm(train_loader)
    logger.info('Freelb training epoch {}...'.format(epoch))
    if adv_steps2 == None:
        adv_steps2 = adv_steps * 2
    if adv_lr2 == None:
        adv_lr2 = adv_lr
    for model_inputs, labels_idx in pbar:
        labels = torch.tensor([i[0] for i in labels_idx])
        indices = torch.tensor([i[1] for i in labels_idx])
        selected_set = set(result_data_indices)
        data_selected = torch.tensor([1 if int(i) in selected_set else 0 for i in indices])
        data_not_selected = torch.tensor([0 if int(i) in selected_set else 1 for i in indices])

        if save_steps > 0 and global_step % save_steps == 0:
            save_model_one_step(args, global_step, model, output_dir=args.output_dir, tokenizer=tokenizer)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()

        # adv strength 1
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)  # 对embedding做mask？
            input_lengths = torch.sum(input_mask, 1)
            if adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)  # 有些会被mask掉，所以乘以* input_mask.unsqueeze(2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
                                                                  adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)

        total_loss = 0.0
        for astep in range(adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            # logits = model(**batch).logits # todo ?不确定
            logits = model(**batch, return_dict=False)[0]  # todo ?不确定
            # _, preds = logits.max(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            # loss = loss / adv_steps
            total_loss += loss.item()
            loss.backward()
            # loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()

            model.zero_grad()
            optimizer.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
        # tr_loss += total_loss

        delta.requires_grad = False
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        model.zero_grad()
        optimizer.zero_grad()
        adv_strength_1_logits = model(**batch).logits
        adv_strength_1_losses = F.cross_entropy(adv_strength_1_logits, labels.squeeze(-1), reduction="none")
        adv_strength_1_probs = F.softmax(adv_strength_1_logits, dim=1)
        adv_strength_1_coef = (1 - torch.tensor([adv_strength_1_probs[i][labels[i]] for i in range(len(labels))]))

        # adv strength 2
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)  # 对embedding做mask？
            input_lengths = torch.sum(input_mask, 1)
            if adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
                                                                  adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)

        total_loss = 0.0
        for astep in range(adv_steps2):
            # (0) forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            # logits = model(**batch).logits #
            logits = model(**batch, return_dict=False)[0]  #
            # _, preds = logits.max(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            # loss = loss / adv_steps
            total_loss += loss.item()
            loss.backward()
            # loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr2 * delta_grad / denorm).detach()
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr2 * delta_grad / denorm).detach()

            model.zero_grad()
            optimizer.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
        # tr_loss += total_loss

        delta.requires_grad = False
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        model.zero_grad()
        optimizer.zero_grad()
        adv_strength_2_logits = model(**batch).logits
        adv_strength_2_losses = F.cross_entropy(adv_strength_2_logits, labels.squeeze(-1), reduction="none")
        adv_strength_2_probs = F.softmax(adv_strength_2_logits, dim=1)
        adv_strength_2_coef = 1 - adv_strength_1_coef

        robust_loss = adv_strength_1_coef.to(device).mul(adv_strength_1_losses)
        non_robust_loss = adv_strength_2_coef.to(device).mul(adv_strength_2_losses)

        losses = robust_loss + non_robust_loss

        loss = torch.mean(losses)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 对所有参数做，防止梯度过大
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        avg_loss.update(total_loss)
        pbar.set_description(f'epoch: {epoch: d}, '
                             f'loss: {avg_loss.get_metric(): 0.4f}, '
                             f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
        global_step += 1

    return global_step


class IndexedDataset(arrow_dataset.Dataset):
    def __init__(self, arrow_table, info=None, split=None, indices_table=None, fingerprint=None):
        super().__init__(arrow_table, info=None, split=None, indices_table=None, fingerprint=None)

    def _getitem(self, key, **kwargs):
        """
        Can be used to index columns (by string names) or rows (by integer, slice, or list-like of integer indices)
        """
        if isinstance(key, bool):
            raise TypeError("dataset index must be int, str, slice or collection of int, not bool")
        format_type = kwargs["format_type"] if "format_type" in kwargs else self._format_type
        format_columns = kwargs["format_columns"] if "format_columns" in kwargs else self._format_columns
        output_all_columns = (
            kwargs["output_all_columns"] if "output_all_columns" in kwargs else self._output_all_columns
        )
        format_kwargs = kwargs["format_kwargs"] if "format_kwargs" in kwargs else self._format_kwargs
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        formatter = datasets.formatting.get_formatter(format_type, features=self._info.features, **format_kwargs)
        pa_subtable = datasets.formatting.query_table(self._data, key, indices=self._indices if self._indices is not None else None)
        formatted_output = datasets.formatting.format_table(
            pa_subtable, key, formatter=formatter, format_columns=format_columns, output_all_columns=output_all_columns
        )
        return (formatted_output, key)

    def __getitems__(self, keys):
        """Can be used to get a batch using a list of integers indices."""
        batch, idx = self.__getitem__(keys)
        # print(f"batch in getitems {batch}")
        # print(f'idx {idx}')
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)], idx


def train_one_epoch_new(args, avg_loss, device, epoch, model, optimizer, scheduler, train_dataset, global_step,
                        save_steps, tokenizer, result_data_indices, beta=1, b=0.1):
    train_dataset = IndexedDataset(train_dataset._data)
    train_dataset = train_dataset.with_transform(preprocess_train)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=indexed_collate)

    pbar = tqdm(train_loader, ncols=100)
    for model_inputs, labels_idx in pbar:
        # print('model inputs')
        # print(model_inputs)
        labels = labels_idx[0]
        indices = labels_idx[1]
        selected_set = set(result_data_indices)
        data_selected = torch.tensor([1 if int(i) in selected_set else 0 for i in indices])
        data_not_selected = torch.tensor([0 if int(i) in selected_set else 1 for i in indices])
        if save_steps > 0 and global_step % save_steps == 0:
            save_model_one_step(args, global_step, model, output_dir=args.output_dir, tokenizer=tokenizer)
        batch_loss = 0
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        logits = model(**model_inputs, return_dict=False)[0]
        _, preds = logits.max(dim=-1)

        # losses_soft = SoftCrossEntropy(logits,labels.squeeze(-1),reduction="none",soft_label=args.soft_label,device=device)
        losses_flooded = (F.cross_entropy(logits, labels.squeeze(-1), reduction="none") - b).abs() + b
        losses_hard = F.cross_entropy(logits, labels.squeeze(-1), reduction="none")
        probs = F.softmax(logits, dim=1)

        # non_flooding_coef = beta * (1 - torch.tensor([probs[i][labels[i]] for i in range(len(labels))]))
        non_flooding_coef = beta * (torch.tensor([1 for i in range(len(labels))]))

        flooding_loss = data_selected.to(device).mul(losses_flooded)
        non_flooding_loss = data_not_selected.to(device).mul(non_flooding_coef.to(device)).mul(losses_hard)

        # continue
        losses = flooding_loss + non_flooding_loss
        loss = torch.mean(losses)
        # loss2  = model(**model_inputs,return_dict=False)
        batch_loss = loss.item()
        # loss.backward()
        accelerator.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        avg_loss.update(batch_loss)
        pbar.set_description(f'epoch: {epoch: d}, '
                             f'loss: {avg_loss.get_metric(): 0.4f}, '
                             f'lr: {optimizer.param_groups[0]["lr"]: .3e},'

                             )
        global_step += 1
    writer.add_scalar(f'{args.exp}/Train/Loss', avg_loss.get_metric(), epoch)
    return global_step


def SoftCrossEntropy(inputs, target, reduction='none', soft_label=1.0, device="cuda"):
    """
    soft label loss
    :param inputs:
    :param target:
    :param reduction:
    :param soft_label: golden label的值，剩下的值被其他标签平分
    :param device:
    :return:
    """
    log_likelihood = -F.log_softmax(inputs, dim=1)
    num_labels = inputs.shape[1]
    batch = inputs.shape[0]

    new_target = F.one_hot(target, num_labels).to(device)
    inverse_target = (torch.ones(inputs.shape).to(device) - new_target).to(device)

    new_target = new_target * soft_label + inverse_target * ((1 - soft_label) / (num_labels - 1))
    losses = torch.sum(torch.mul(log_likelihood, new_target), dim=1)
    if reduction == 'average':
        losses = torch.sum(losses) / batch
    elif reduction == "none":
        return losses
    elif reduction == "sum":
        losses = torch.sum(losses)

    return losses


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    finetune(args)
