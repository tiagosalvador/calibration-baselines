from config import *
import os
import tqdm
import torchvision
import wget
import tarfile

# Download CIFAR-10
# dataset = torchvision.datasets.CIFAR10(root=os.path.join(root_folder_datasets, 'cifar10'), train=True, download=True);
# dataset = torchvision.datasets.CIFAR10(root=os.path.join(root_folder_datasets, 'cifar10'), train=False, download=True);

# Download CIFAR-100
# dataset = torchvision.datasets.CIFAR100(root=os.path.join(root_folder_datasets, 'cifar100'), train=True, download=True);
# dataset = torchvision.datasets.CIFAR100(root=os.path.join(root_folder_datasets, 'cifar100'), train=False, download=True);

# Download SVHN
# dataset = torchvision.datasets.SVHN(root=os.path.join(root_folder_datasets, 'svhn'), split='train', download=True);
# dataset = torchvision.datasets.SVHN(root=os.path.join(root_folder_datasets, 'svhn'), split='test', download=True);

# Download CIFAR-10-C
# os.makedirs(os.path.join(root_folder_datasets, desc='Extracting CIFAR-10-C', 'cifar10-c'), exist_ok=True)
# url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar10-c', 'cifar10-c.tar'))
# with tarfile.open(name=os.path.join(root_folder_datasets, 'cifar10-c', 'cifar10-c.tar')) as tar:
#     # Go over each member
#     for member in tqdm.tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
#         # Extract member
#         member.name = member.name.replace('CIFAR-10-C', 'cifar10-c')
#         tar.extract(path=os.path.join(root_folder_datasets), member=member)


# Download CIFAR-100-C
# os.makedirs(os.path.join(root_folder_datasets, 'cifar100-c'), exist_ok=True)
# url = 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar100-c', 'cifar100-c.tar'))
# with tarfile.open(name=os.path.join(root_folder_datasets, 'cifar100-c', 'cifar100-c.tar')) as tar:
#     # Go over each member
#     for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting CIFAR-100-C', total=len(tar.getmembers())):
#         # Extract member
#         member.name = member.name.replace('CIFAR-100-C', 'cifar100-c')
#         tar.extract(path=os.path.join(root_folder_datasets), member=member)

# Download STL10
# dataset = torchvision.datasets.STL10(root=os.path.join(root_folder_datasets, 'stl10'), split='train', download=True);
# dataset = torchvision.datasets.STL10(root=os.path.join(root_folder_datasets, 'stl10'), split='test', download=True);

# Download CIFAR10.1 v4
# os.makedirs(os.path.join(root_folder_datasets, 'cifar10.1-v4'), exist_ok=True)
# url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v4', 'cifar10.1_v4_data.npy'))
# url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v4', 'cifar10.1_v4_labels.npy'))

# Download CIFAR10.1 v6
# os.makedirs(os.path.join(root_folder_datasets, 'cifar10.1-v6'), exist_ok=True)
# url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v6', 'cifar10.1_v6_data.npy'))
# url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy'
# filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v6', 'cifar10.1_v6_labels.npy'))