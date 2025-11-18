# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import sys
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import numpy as np
import math

from PIL import Image
from collections import Counter
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_labeled_unlabeled_data


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

def get_food101(args, name, data_dir='./data', include_lb_to_ulb=False):
    # 
    cache_dir = os.path.join(data_dir, 'food101_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 
    cache_file = os.path.join(cache_dir, f'food101_{args.img_size}.pkl')
    
    # 
    if os.path.exists(cache_file):
        print(f"Loading cached Food101 data from {cache_file}")
        with open(cache_file, 'rb') as f:
            if sys.version_info[0] == 2:
                cache_data = pickle.load(f)
            else:
                cache_data = pickle.load(f, encoding='latin1')
        
        train_data = cache_data['train_data']
        train_targets = cache_data['train_targets']
        test_data = cache_data['test_data']
        test_targets = cache_data['test_targets']
    else:
        # 
        print("Creating cache for Food101 dataset...")
        data_dir = os.path.join(data_dir, name.lower())
        dset = getattr(torchvision.datasets, 'Food101')
        train_dset = dset(data_dir, split='train', download=True)
        test_dset = dset(data_dir, split='test', download=True)
        
        # 
        train_data = []
        train_targets = []
        time = 0
        print("Processing training data...")
        for img_path, target in zip(train_dset._image_files, train_dset._labels):
            img = Image.open(img_path).convert("RGB")
            if args.img_size != img.size[0]:  # 
                img = img.resize((args.img_size, args.img_size))
            img_array = np.array(img)
            train_data.append(img_array)
            train_targets.append(target)
            
            time +=1
            if time % 1000 == 0:
                print(f"Processing training data...{time}")
        
        # 
        test_data = []
        test_targets = []
        print("Processing test data...")
        for img_path, target in zip(test_dset._image_files, test_dset._labels):
            img = Image.open(img_path).convert("RGB")
            if args.img_size != img.size[0]: 
                img = img.resize((args.img_size, args.img_size))
            img_array = np.array(img)
            test_data.append(img_array)
            test_targets.append(target)
        
        # (H, W, C) -> (C, H, W)
        train_data = np.array(train_data).transpose((0, 3, 1, 2))
        train_targets = np.array(train_targets)
        test_data = np.array(test_data).transpose((0, 3, 1, 2))
        test_targets = np.array(test_targets)
        
        # 
        cache_data = {
            'train_data': train_data,
            'train_targets': train_targets,
            'test_data': test_data,
            'test_targets': test_targets
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Food101 cache saved to {cache_file}")
    
    # 
    train_data_img, train_targets_img, test_data_img, test_targets_img = get_imagenet(args, 'imagenet', data_dir='./data')
    
    # 
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    # 
    lb_idx, ulb_idx, ulb_ood_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(
        args, train_data, train_targets, train_data_img, train_targets_img,
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
        include_lb_to_ulb=include_lb_to_ulb
    )
    
    # (C, H, W) -> (H, W, C) 
    data, targets, noised_targets = np.array(train_data).transpose((0, 2, 3, 1)), np.array(train_targets), np.array(train_targets)
    train_data_img, train_targets_img = np.array(train_data_img), np.array(train_targets_img)
    
    # 
    lb_count = [0 for _ in range(args.num_classes)]
    lb_clean_count = [0 for _ in range(args.num_classes)]
    lb_noise_count = [0 for _ in range(args.num_classes)]
    new_lb_noise_count = [0 for _ in range(args.num_classes)]
    ulb_count = [0 for _ in range(args.num_classes + args.num_ood_classes)]

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

    # 
    merged_data = np.concatenate([data, train_data_img], axis=0)
    merged_targets = np.concatenate([targets, train_targets_img], axis=0)
    adjusted_ulb_idx = ulb_ood_idx + len(data)
    merged_indices = np.concatenate([ulb_idx, adjusted_ulb_idx])

    # 
    test_data_processed = test_data.transpose((0, 2, 3, 1)) if len(test_data.shape) == 4 else test_data

    lb_dset = BasicDataset(lb_idx, data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(merged_indices, merged_data[merged_indices], merged_targets[merged_indices], None, 
                           (args.num_classes+args.num_ood_classes), True,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(None, test_data_processed, test_targets, None, args.num_classes, False, 
                           weak_transform=transform_val, strong_transform=None, onehot=False)

    lb_count_message = {
        'lb_count': lb_count, 
        'ulb_count': ulb_count, 
        'lb_clean_count': lb_clean_count,
        'lb_noise_count': lb_noise_count, 
        'new_lb_noise_count': new_lb_noise_count
    }
    
    return data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message

def get_food101_old(args, name, data_dir='./data', include_lb_to_ulb=False):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, 'Food101')
    train_dset = dset(data_dir, split='train', download=True)
    test_dset = dset(data_dir, split='test', download=True)
    
    #train_data, train_targets = train_dset._image_files, train_dset._labels
    #test_data, test_targets = test_dset._image_files, test_dset._labels
    train_data = []
    train_targets = []
    for img_path, target in zip(train_dset._image_files, train_dset._labels):
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((args.img_size, args.img_size))  
        img_array = np.array(img)
        train_data.append(img_array)
        train_targets.append(target)
    
    test_data = []
    test_targets = []
    for img_path, target in zip(test_dset._image_files, test_dset._labels):
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((args.img_size, args.img_size))  
        img_array = np.array(img)
        test_data.append(img_array)
        test_targets.append(target)
    
    # 
    train_data = np.array(train_data)
    train_targets = np.array(train_targets)
    test_data = np.array(test_data)
    test_targets = np.array(test_targets)
    
    train_data_img, train_targets_img, test_data_img, test_targets_img = get_imagenet(args, 'imagenet', data_dir='./data')
    
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    #lb_data, lb_targets, ulb_data, ulb_targets = split_labeled_unlabeled_data(args, data, targets, num_classes,
    #                                                            lb_num_labels=num_labels,
    #                                                            ulb_num_labels=args.ulb_num_labels,
    #                                                            lb_imbalance_ratio=args.lb_imb_ratio,
    #                                                            ulb_imbalance_ratio=args.ulb_imb_ratio,
    #                                                            include_lb_to_ulb=include_lb_to_ulb)
                                                                
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

    # 1.
    merged_data = np.concatenate([data, train_data_img], axis=0)
    merged_targets = np.concatenate([targets, train_targets_img], axis=0)

    # 2.
    adjusted_ulb_idx = ulb_ood_idx + len(data)
    merged_indices = np.concatenate([ulb_idx, adjusted_ulb_idx])

    # merged_test_data = np.concatenate([test_data_img, test_data], axis=0)
    # merged_test_targets = np.concatenate([test_targets_img, test_targets], axis=0)

    lb_dset = BasicDataset(lb_idx, data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(merged_indices, merged_data[merged_indices], merged_targets[merged_indices], None, (args.num_classes+args.num_ood_classes), True,
                            weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(None, test_data, test_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)

    lb_count_message = {'lb_count': lb_count, 'ulb_count': ulb_count, 'lb_clean_count': lb_clean_count,
                        'lb_noise_count': lb_noise_count, 'new_lb_noise_count': new_lb_noise_count}
    
    #lb_count = [0 for _ in range(num_classes)]
    #ulb_count = [0 for _ in range(num_classes)]
    #for c in lb_targets:
    #    lb_count[c] += 1
   # for c in ulb_targets:
   #     ulb_count[c] += 1
   # print("lb count: {}".format(lb_count))
   # print("ulb count: {}".format(ulb_count))

   # if alg == 'fullysupervised':
   #    lb_data = data
   #     lb_targets = targets

    #lb_dset = Food101Dataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    #ulb_dset = Food101Dataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    #dset = getattr(torchvision.datasets,'Food101')
    #dset = dset(data_dir, split='test', download=True)
    #test_data, test_targets = dset._image_files, dset._labels
    #eval_dset = Food101Dataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    #return lb_dset, ulb_dset, eval_dset
    return data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message

#class Food101Dataset(BasicDataset):
#    def __sample__(self, idx):
#        path = self.data[idx]
 #       img = Image.open(path).convert("RGB")
#       target = self.targets[idx]
#        return img, target 

