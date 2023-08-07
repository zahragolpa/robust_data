import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import logging
import sys

import utils as utils
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from datasets import load_dataset
from transformers import (
    AdamW, AutoConfig, AutoTokenizer, AutoFeatureExtractor, ViTForImageClassification, ViTFeatureExtractor
)

from tqdm import tqdm
import cv2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['google/vit-base-patch16-224-in21k']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='cifar10',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-m', '--model-name', metavar='ARCH', default='google/vit-base-patch16-224-in21k',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained',
                    #action='store_true',
                    default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

# robust data statistics
parser.add_argument('--statistic_interval', default=1, type=float, help='')
parser.add_argument('--dataset_len', default=None, type=int, help='')
parser.add_argument('--with_untrained_model', default=1, type=int, help='')
parser.add_argument('--use_cur_preds', default=0, type=int, help='whether use cur predictions or golden labels to calculate loss')
parser.add_argument('--do_train_shuffle', default=1, type=int, help='')

best_acc1 = 0
train_len = 0


def robust_statistics_perturbation(model, train_dev_loader, train_set_len, device, use_cur_preds=True):
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics = {}
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
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["golden_label"] = cur_label.item()
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item() == cur_pred.item())
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
        # attention_mask = model_inputs['attention_mask']

        # embedding_init = word_embedding_layer(input_ids)

        transformed_batch = torch.empty_like(input_ids)
        for i in range(input_ids.shape[0]):
            blurred = cv2.blur(input_ids[i].permute(1, 2, 0).detach().cpu().numpy(), (5, 5))
            transformed_batch[i] = torch.from_numpy(blurred).permute(2, 0, 1)

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
        # loss = loss / args.adv_steps
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


def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    output_dict = {"pixel_values": pixel_values}
    return output_dict, labels


def preprocess_dev(example_batch):
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768))
    dev_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    example_batch["pixel_values"] = [dev_transforms(image.convert("RGB")) for image in example_batch["img"]]
    return example_batch


def preprocess_train(example_batch):
    """Apply _train_transforms across a batch."""
    size = (224, 224)
    normalize = transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768))
    train_transforms = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
    return example_batch


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    # print(ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model_name))
        if args.model_name == 'google/vit-base-patch16-224-in21k':
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            model = models.__dict__[args.model_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model_name))
        if args.model_name == 'google/vit-base-patch16-224-in21k':
            model = ViTForImageClassification()
        else:
            model = models.__dict__[args.model_name]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model_name.startswith('alexnet') or args.model_name.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())

    elif args.data == 'cifar10':
        args.num_labels = 10
        if args.model_name == 'google/google/vit-base-patch16-224-in21k':
            tokenizer = AutoFeatureExtractor.from_pretrained(args.model_name)
        train_dataset = load_dataset('cifar10', split='train')
        train_dataset = train_dataset.with_transform(preprocess_train)

        test_dataset = load_dataset('cifar10', split='test')
        test_dataset = test_dataset.with_transform(preprocess_dev)

        logger.info("train dataset length: "+ str(len(train_dataset)))
        # train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=args.do_train_shuffle
        #                           , collate_fn=collate_fn, sampler=train_sampler
        #                           )
        # train_dev_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collate_fn)
        # dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collate_fn, sampler=dev_sampler)
        logger.info("dev dataset length: "+ str(len(test_dataset)))

        # if args.do_test:
            #test_dataset = utils.VisionDataset(args, name_or_dataset=args.dataset_name, split='test')
            #test_dataset = CIFAR10(data, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
            # test_dataset = load_dataset('cifar10', split='test')
            # test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator, sampler=dev_sampler)

        train_dev_dict = {}
        labels = train_dataset.features["label"].names
        label2id = {label: str(i) for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(args.model_name, num_labels=len(labels), i2label=id2label, label2id=label2id, finetuning_task="image-classification")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        model.to(device)
        print(model)

    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    global train_len
    train_len = len(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler)

    train_dev_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, 1, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, train_dev_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)


def train(train_loader, train_dev_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    global train_len

    epoch_steps = train_len // args.batch_size
    global_step = 0
    robust_statistics_dict = {}

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # global train_len
#        if args.with_untrained_model:
#            if global_step % int(args.statistic_interval * epoch_steps) == 0:
#                cur_robust_statistics = robust_statistics_perturbation(model, train_dev_loader,
#                                                                       train_set_len=train_len, device=device,
#                                                                       use_cur_preds=args.use_cur_preds)
#                robust_statistics_dict[global_step] = cur_robust_statistics
#
#        else:
#            if global_step % int(args.statistic_interval * epoch_steps) == 0 and global_step != 0:
#                cur_robust_statistics = robust_statistics_perturbation(model, train_dev_loader,
#                                                                       train_set_len=train_len, device=device,
#                                                                       use_cur_preds=args.use_cur_preds)
#                robust_statistics_dict[global_step] = cur_robust_statistics
#
        # move data to the same device as model
        # images = images.to(device, non_blocking=True)
        images = {k: v.to(device) for k, v in images.items()}
        images = images['pixel_values']
        target = target.to(device, non_blocking=True)
        # print(len(images))
        # print(type(images))
        # print(type(images['pixel_values']))
        # print(**images)

        # compute output
        output = model(images, return_dict=False)[0]
        # print(output)
        # output = model(**images, return_dict=False)[0]
        # print(f"output {len(output)}, target {len(target.squeeze(-1))}")
        loss = criterion(output, target.squeeze(-1))
        # print(f'loss {loss} mean {torch.mean(loss)}')

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            niter = epoch*len(train_loader)+i
            writer.add_scalar('Train/Loss', losses.val, niter)
            writer.add_scalar('Train/Acc@1', top1.val, niter)
            writer.add_scalar('Train/Acc@5', top5.val, niter)


def validate(val_loader, model, criterion, epoch, args):

    def run_validate(loader, epoch, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                images = {k: v.to(torch.device("cuda")) for k, v in images.items()}
                images = images['pixel_values']
                i = base_progress + i
                # if args.gpu is not None and torch.cuda.is_available():
                    # images = images.cuda(args.gpu, non_blocking=True)
                    # images = {k: v.to(torch.device("cuda")) for k, v in images.items()}
                    # images = {k: v.to(torch.device("cuda")) for k, v in images.items()}
                    # images = images['pixel_values']
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images, return_dict=False)[0]
                loss = criterion(output, target.squeeze(-1))

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
                    niter = epoch*len(val_loader)+i
                    writer.add_scalar('Test/Loss', losses.val, niter)
                    writer.add_scalar('Test/Acc@1', top1.val, niter)
                    writer.add_scalar('Test/Acc@5', top5.val, niter)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader, epoch)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, epoch, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
