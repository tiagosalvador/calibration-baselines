root_folder_datasets = 'datasets'

import os

import torch
import torchvision.transforms as transforms

transform_cifar10 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_stl10 = transforms.Compose(
    [
     transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
def load_ds_info(dataset, net=None):
    if dataset == 'cifar10':
        ds_info = {
            'name': 'cifar10',
            'num_classes': 10,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar10')
        }
    elif dataset == 'cifar10-c':
        ds_info = {
            'name': 'cifar10-c',
            'num_classes': 10,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar10-c')
        }
    elif dataset == 'cifar100':
        ds_info = {
            'name': 'cifar100',
            'num_classes': 100,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar100')
        }
    elif dataset == 'cifar100-c':
        ds_info = {
            'name': 'cifar100-c',
            'num_classes': 100,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar100-c')
        }
    elif dataset == 'svhn':
        ds_info = {
            'name': 'svhn',
            'num_classes': 10,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'svhn')
        }
    elif dataset == 'stl10':
        ds_info = {
            'name': 'stl10',
            'num_classes': 10,
            'transform': transform_stl10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'stl10')
        }
    elif dataset == 'cifar10.1-v4':
        ds_info = {
            'name': 'cifar10.1-v4',
            'num_classes': 10,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar10.1-v4')
        }
    elif dataset == 'cifar10.1-v6':
        ds_info = {
            'name': 'cifar10.1-v6',
            'num_classes': 10,
            'transform': transform_cifar10,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'cifar10.1-v6')
        }
    elif dataset == 'imagenet':
        ds_info = {
            'name': 'imagenet',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 512,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet')
        }
    elif dataset == 'imagenet-c':
        ds_info = {
            'name': 'imagenet-c',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-c')
        }
    elif dataset == 'imagenet-v2-mf':
        ds_info = {
            'name': 'imagenet-v2-mf',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-v2-mf')
        }
    elif dataset == 'imagenet-v2-thr':
        ds_info = {
            'name': 'imagenet-v2-thr',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-v2-thr')
        }
    elif dataset == 'imagenet-v2-ti':
        ds_info = {
            'name': 'imagenet-v2-ti',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-v2-ti')
        }
    elif dataset == 'imagenet-sketch':
        ds_info = {
            'name': 'imagenet-sketch',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-sketch')
        }
    elif dataset == 'imagenet-a':
        ds_info = {
            'name': 'imagenet-a',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-a')
        }
    elif dataset == 'imagenet-r':
        ds_info = {
            'name': 'imagenet-r',
            'num_classes': 1000,
            'transform': transform_imagenet,
            'batch_size': 256,
            'root_folder_datasets': os.path.join(root_folder_datasets, 'imagenet-r')
        }
    if net is not None:
        ds_info['architecture'] = net.architecture    
        ds_info['dim_features'] = net.dim_features
    return ds_info