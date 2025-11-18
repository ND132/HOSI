# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import json
import random
import torchvision
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from collections import Counter

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]

def get_imagenet_ori(args, name, data_dir='./data', include_lb_to_ulb=False):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    # if args.augment_type == "RandAugment":
    #     My_RandAugment = RandAugment
    # else:
    #     My_RandAugment = RandAugmentPlus

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    if img_size == 32:
        data_dir = os.path.join(data_dir, 'imagenet127')
    else:
        data_dir = os.path.join(data_dir, 'imagenet_127_64')

    train_data = []
    train_targets = []

    for filename in train_list:
        file = os.path.join(data_dir, filename)
        with open(file, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            train_data.append(entry['data'])
            train_targets.extend(entry['labels'])
    train_targets = [i - 1 for i in train_targets]
    train_data = np.vstack(train_data).reshape((len(train_targets), 3, img_size, img_size))
    train_data = train_data.transpose((0, 2, 3, 1))

    test_data = []
    test_targets = []

    for filename in test_list:
        file = os.path.join(data_dir, filename)
        with open(file, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            test_data.append(entry['data'])
            test_targets.extend(entry['labels'])
    test_targets = [i - 1 for i in test_targets]
    test_data = np.vstack(test_data).reshape((len(test_targets), 3, img_size, img_size))
    test_data = test_data.transpose((0, 2, 3, 1))

    count = Counter(train_targets)
    sorted_classes = sorted(count.keys(), key=lambda cls: (-count[cls], cls))
    class_mapping = {cls: idx for idx, cls in enumerate(sorted_classes)}

    train_targets = [class_mapping[cls] for cls in train_targets]
    test_targets = [class_mapping[cls] for cls in test_targets]

    return train_data, train_targets, test_data, test_targets

def get_imagenet(args, name, data_dir='./data', include_lb_to_ulb=False):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    if img_size == 32:
        data_dir = os.path.join(data_dir, 'imagenet127')
    else:
        data_dir = os.path.join(data_dir, 'imagenet_127_64')

    # Load raw data
    train_data = []
    train_targets = []
    for filename in train_list:
        file = os.path.join(data_dir, filename)
        with open(file, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            train_data.append(entry['data'])
            train_targets.extend(entry['labels'])
    train_targets = [i - 1 for i in train_targets]
    train_data = np.vstack(train_data).reshape((len(train_targets), 3, img_size, img_size))
    train_data = train_data.transpose((0, 2, 3, 1))

    test_data = []
    test_targets = []
    for filename in test_list:
        file = os.path.join(data_dir, filename)
        with open(file, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            test_data.append(entry['data'])
            test_targets.extend(entry['labels'])
    test_targets = [i - 1 for i in test_targets]
    test_data = np.vstack(test_data).reshape((len(test_targets), 3, img_size, img_size))
    test_data = test_data.transpose((0, 2, 3, 1))

    # Count class frequencies and sort
    count = Counter(train_targets)
    sorted_classes = sorted(count.keys(), key=lambda cls: (-count[cls], cls))
    all_classes = list(set(train_targets))

    # Randomly select num_ood_classes classes
    if hasattr(args, 'num_ood_classes') and args.num_ood_classes > 0:
        selected_classes = random.sample(sorted_classes, args.num_ood_classes)
        #selected_classes = random.sample(all_classes, args.num_ood_classes)
        #selected_classes = [13,18,35,67,104,0,22,33,44,55]######################################################################
        #selected_classes = [2,21,51,85,101,0,22,33,44,55]
    else:
        selected_classes = sorted_classes  # Default to all classes

    # Filter data to only include selected classes
    train_mask = np.isin(train_targets, selected_classes)
    test_mask = np.isin(test_targets, selected_classes)

    train_data = train_data[train_mask]
    train_targets = np.array(train_targets)[train_mask]
    test_data = test_data[test_mask]
    test_targets = np.array(test_targets)[test_mask]

    # Remap labels to 0,1,...,num_ood_classes-1
    class_mapping = {cls: idx+args.num_classes for idx, cls in enumerate(selected_classes)}
    train_targets = [class_mapping[cls] for cls in train_targets]
    test_targets = [class_mapping[cls] for cls in test_targets]

    return train_data, train_targets, test_data, test_targets

def get_cifar(args, name, data_dir='./data', include_lb_to_ulb=False):
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())

    train_dset = dset(data_dir, train=True, download=True)
    test_dset = dset(data_dir, train=False, download=True)

    train_data, train_targets = train_dset.data, train_dset.targets
    test_data, test_targets = test_dset.data, test_dset.targets

    train_data_img, train_targets_img, test_data_img, test_targets_img = get_imagenet(args, 'imagenet', data_dir='./data')

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    lb_idx, ulb_idx, ulb_ood_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(args, train_data, train_targets,
                                                                               train_data_img, train_targets_img,
                                                                               num_classes=args.num_classes,
                                                                               num_ood_classes=args.num_ood_classes,
                                                                               lb_num_labels=args.num_labels,
                                                                               ulb_num_labels=args.ulb_num_labels,
                                                                               lb_imbalance_ratio=args.lb_imb_ratio,
                                                                               ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                               noise_ratio=args.noise_ratio,
                                                                               noise_per_class=args.noise_per_class,
                                                                               lb_imb_type=args.lb_imb_type,
                                                                               ulb_imb_type=args.ulb_imb_type,
                                                                               num_steps=args.num_steps,
                                                                               include_lb_to_ulb=include_lb_to_ulb)

    data, targets, noised_targets = np.array(train_data), np.array(train_targets), np.array(train_targets)
    train_data_img, train_targets_img = np.array(train_data_img), np.array(train_targets_img)
    lb_count = [0 for _ in range(args.num_classes)]
    lb_clean_count = [0 for _ in range(args.num_classes)]
    lb_noise_count = [0 for _ in range(args.num_classes)]
    new_lb_noise_count = [0 for _ in range(args.num_classes)]
    ulb_count = [0 for _ in range(args.num_classes + args.num_ood_classes)]

    print("lb_clean_idx 类型:", type(lb_clean_idx))  # 如果是numpy数组
    print("lb_clean_idx 数据类型:", lb_clean_idx.dtype)  # 检查dtype
    print("lb_clean_idx 示例值:", lb_clean_idx[:5])  # 查看前5个值

    for c in targets[lb_idx]:
        lb_count[c] += 1
    for c in targets[ulb_idx]:
        ulb_count[c] += 1
    for c in train_targets_img[ulb_ood_idx]:
        ulb_count[c] += 1
    for c in targets[lb_clean_idx]:
        lb_clean_count[c] += 1
    if len(lb_noise_idx) > 0:
        for c in targets[lb_noise_idx]:
            lb_noise_count[c] += 1

    p_noise = np.zeros((args.num_classes, args.num_classes))
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            if i != j:
                p_noise[i][j] = lb_count[j] / (sum(lb_count) - lb_count[i])
    p_noise = p_noise / p_noise.sum(axis=1, keepdims=True)

    for i in lb_noise_idx:
        if args.noise_type == 'sym':
            noised_targets[i] = (random.randint(1, args.num_classes - 1) + targets[i]) % args.num_classes
        elif args.noise_type == 'asym':
            noised_targets[i] = np.random.choice(args.num_classes, p=p_noise[targets[i]])
        elif args.noise_type == 'circle':
            noised_targets[i] = (targets[i] + 1) % args.num_classes
    if len(lb_noise_idx) > 0:
        for c in noised_targets[lb_noise_idx]:
            new_lb_noise_count[c] += 1

    print("lb count: {}".format(lb_count))
    print("lb clean count: {}".format(lb_clean_count))
    print("lb noise count: {}".format(lb_noise_count))
    print("new lb noise count: {}".format(new_lb_noise_count))
    print("ulb count: {}".format(ulb_count))

    # 1. 合并数据和标签
    merged_data = np.concatenate([data, train_data_img], axis=0)
    merged_targets = np.concatenate([targets, train_targets_img], axis=0)

    # 2. 合并索引（调整 ulb_idx）
    adjusted_ulb_idx = ulb_ood_idx + len(data)
    merged_indices = np.concatenate([ulb_idx, adjusted_ulb_idx])

    # merged_test_data = np.concatenate([test_data_img, test_data], axis=0)
    # merged_test_targets = np.concatenate([test_targets_img, test_targets], axis=0)

    lb_dset = BasicDataset(lb_idx, data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(ulb_idx, data[ulb_idx], targets[ulb_idx], None, args.num_classes, True,
                            weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(None, test_data, test_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)

    lb_count_message = {'lb_count': lb_count, 'ulb_count': ulb_count, 'lb_clean_count': lb_clean_count,
                        'lb_noise_count': lb_noise_count, 'new_lb_noise_count': new_lb_noise_count}

    return data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message
