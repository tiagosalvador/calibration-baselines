from pytorchcv.model_provider import get_model as ptcv_get_model
from tqdm import tqdm
from imgclsmob.pytorch.utils import prepare_model as prepare_model_pt

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

from scipy.optimize import minimize
from sklearn import linear_model
import scipy.stats

from uncertainty_measures import get_uncertainty_measures

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def small_large_split(y_data, nsamples, ds_info):
    indices_small = np.zeros(len(y_data), dtype=bool)
    indices_large = np.zeros(len(y_data), dtype=bool)
    for y in range(ds_info['num_classes']):
        select = y_data==y
        indices_y = np.arange(len(y_data))[select]
        choice = np.random.choice(range(len(indices_y)), size=nsamples//ds_info['num_classes'], replace=False)    
        ind = np.zeros(len(indices_y), dtype=bool)
        ind[choice] = True
        rest = ~ind
        indices_small[indices_y[ind]] = True
        indices_large[indices_y[rest]] = True
    return indices_small, indices_large


def get_loader(subset, ds_info, shuffle=False):
    if subset != 'train':
        indices = ds_info['indices_'+subset]
    if ds_info['name'] == 'cifar10':
        transform = ds_info['transform']
        dataset = torchvision.datasets.CIFAR10(root=ds_info['root_folder_datasets'], train=subset=='train',
                                               download=False, transform=transform);
        if subset != 'train':
            dataset.data = dataset.data[indices]
            dataset.targets = np.array(dataset.targets)[indices]
    elif ds_info['name'] == 'cifar100':
        transform = ds_info['transform']
        dataset = torchvision.datasets.CIFAR100(root=ds_info['root_folder_datasets'], train=subset=='train',
                                               download=False, transform=transform);
        if subset != 'train':
            dataset.data = dataset.data[indices]
            dataset.targets = np.array(dataset.targets)[indices]
    elif ds_info['name'] == 'svhn':
        transform = ds_info['transform']
        split = 'test' if subset == 'cal' else subset
        dataset = torchvision.datasets.SVHN(root=ds_info['root_folder_datasets'], split=split,
                                               download=False, transform=transform);
        if subset != 'train':
            dataset.data = dataset.data[indices]
            dataset.targets = np.array(dataset.labels)[indices]
    elif ds_info['name'] == 'imagenet':
        if subset == 'train':
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
        elif subset == 'train_large':
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
        else:
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'val')
        dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)[indices]
        dataset.targets = np.array(dataset.targets)[indices]
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds_info['batch_size'], shuffle=shuffle, num_workers=4)
    return loader


def get_features_logits_labels(net, subset, ds_info, save=True):
    experiments_folder = ds_info['folder']
    suffix_name = '.npy'
    folder_path = os.path.join(experiments_folder,'features_logits_labels')
    if os.path.exists(os.path.join(folder_path, 'labels_'+subset+suffix_name)):
        features = torch.from_numpy(np.load(os.path.join(folder_path, 'features_'+subset+suffix_name)))
        logits = torch.from_numpy(np.load(os.path.join(folder_path, 'logits_'+subset+suffix_name)))
        labels = torch.from_numpy(np.load(os.path.join(folder_path, 'labels_'+subset+suffix_name)))
    else:
        dataloader = get_loader(subset, ds_info)
        net.eval()
        features = torch.zeros(len(dataloader.dataset.targets), ds_info['dim_features'])
        logits = torch.zeros(len(dataloader.dataset.targets), ds_info['num_classes'])
        start = 0
        end = 0
        with torch.no_grad():
            suffix_message = " for clean"
            for data in tqdm(dataloader, desc=f"Generating features, logits and labels"+suffix_message, leave=False):
                images, labels = data
                end += len(images)
                # compute features and logits
                features_temp = net.feature_extractor(images.cuda())
                logits[start:end] = net.output(features_temp).cpu()
                if 'squeezenet' in ds_info['architecture']:
                    features[start:end] = features_temp.cpu().reshape(features_temp.shape[0],-1)
                else:
                    features[start:end] = features_temp.cpu()            
                start = end
        labels = torch.from_numpy(dataloader.dataset.targets)
        if save:
            np.save(os.path.join(folder_path, 'features_'+subset+suffix_name), features)
            np.save(os.path.join(folder_path, 'logits_'+subset+suffix_name), logits)
            np.save(os.path.join(folder_path, 'labels_'+subset+suffix_name), labels)    

    return features, logits, labels
