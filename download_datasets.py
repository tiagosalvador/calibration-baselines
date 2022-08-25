from config import *
import os
import tqdm
import torchvision
import wget
import tarfile

# Download CIFAR-10
dataset = torchvision.datasets.CIFAR10(root=os.path.join(root_folder_datasets, 'cifar10'), train=True, download=True);
dataset = torchvision.datasets.CIFAR10(root=os.path.join(root_folder_datasets, 'cifar10'), train=False, download=True);

# Download CIFAR-100
dataset = torchvision.datasets.CIFAR100(root=os.path.join(root_folder_datasets, 'cifar100'), train=True, download=True);
dataset = torchvision.datasets.CIFAR100(root=os.path.join(root_folder_datasets, 'cifar100'), train=False, download=True);

# Download SVHN
dataset = torchvision.datasets.SVHN(root=os.path.join(root_folder_datasets, 'svhn'), split='train', download=True);
dataset = torchvision.datasets.SVHN(root=os.path.join(root_folder_datasets, 'svhn'), split='test', download=True);

# Download CIFAR-10-C
os.makedirs(os.path.join(root_folder_datasets, 'cifar10-c'), exist_ok=True)
url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
filename = wget.download(url, out=os.path.join('datasets', 'cifar10-c', 'cifar10-c.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'cifar10-c', 'cifar10-c.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting CIFAR-10-C', total=len(tar.getmembers())):
        # Extract member
        member.name = member.name.replace('CIFAR-10-C', 'cifar10-c')
        tar.extract(path=os.path.join(root_folder_datasets), member=member)


# Download CIFAR-100-C
os.makedirs(os.path.join(root_folder_datasets, 'cifar100-c'), exist_ok=True)
url = 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar'
filename = wget.download(url, out=os.path.join('datasets', 'cifar100-c', 'cifar100-c.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'cifar100-c', 'cifar100-c.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting CIFAR-100-C', total=len(tar.getmembers())):
        # Extract member
        member.name = member.name.replace('CIFAR-100-C', 'cifar100-c')
        tar.extract(path=os.path.join(root_folder_datasets), member=member)

# Download STL10
dataset = torchvision.datasets.STL10(root=os.path.join(root_folder_datasets, 'stl10'), split='train', download=True);
dataset = torchvision.datasets.STL10(root=os.path.join(root_folder_datasets, 'stl10'), split='test', download=True);

# Download CIFAR10.1 v4
os.makedirs(os.path.join(root_folder_datasets, 'cifar10.1-v4'), exist_ok=True)
url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy'
filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v4', 'cifar10.1_v4_data.npy'))
url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy'
filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v4', 'cifar10.1_v4_labels.npy'))

# Download CIFAR10.1 v6
os.makedirs(os.path.join(root_folder_datasets, 'cifar10.1-v6'), exist_ok=True)
url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy'
filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v6', 'cifar10.1_v6_data.npy'))
url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy'
filename = wget.download(url, out=os.path.join('datasets', 'cifar10.1-v6', 'cifar10.1_v6_labels.npy'))


# Download ImageNet-C
os.makedirs(os.path.join(root_folder_datasets, 'imagenet-c'), exist_ok=True)
url = 'https://zenodo.org/record/2235448/files/blur.tar'
filename = wget.download(url, out=os.path.join('datasets', 'imagenet-c', 'blur.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-c', 'blur.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting Blur', total=len(tar.getmembers())):
        # Extract member
        tar.extract(path=os.path.join(os.path.join(root_folder_datasets, 'imagenet-c'), member=member)

url = 'https://zenodo.org/record/2235448/files/digital.tar'
filename = wget.download(url, out=os.path.join('datasets', 'imagenet-c', 'digital.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-c', 'digital.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting Digital', total=len(tar.getmembers())):
        # Extract member
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenet-c'), member=member)

url = 'https://zenodo.org/record/2235448/files/extra.tar'
filename = wget.download(url, out=os.path.join('datasets', 'imagenet-c', 'extra.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-c', 'extra.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting Extra', total=len(tar.getmembers())):
        # Extract member
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenet-c'), member=member)

url = 'https://zenodo.org/record/2235448/files/noise.tar'
filename = wget.download(url, out=os.path.join('datasets', 'imagenet-c', 'noise.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-c', 'noise.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting Noise', total=len(tar.getmembers())):
        # Extract member
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenet-c'), member=member)

url = 'https://zenodo.org/record/2235448/files/weather.tar'
filename = wget.download(url, out=os.path.join('datasets', 'imagenet-c', 'weather.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-c', 'weather.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting Weather', total=len(tar.getmembers())):
        # Extract member
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenet-c'), member=member)

                    
# Download ImageNet-v2
os.makedirs(os.path.join(root_folder_datasets, 'imagenetv2-mf'), exist_ok=True)
url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz'
filename = wget.download(url, out=os.path.join('datasets', 'imagenetv2-mf', 'imagenetv2-mf.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenet-v2-mf', 'imagenetv2-mf.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting ImageNet-V2-MF', total=len(tar.getmembers())):
        # Extract member
        member.name = member.name.replace('imagenetv2-matched-frequency-format-val/', '')
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenet-v2-mf'), member=member)

os.makedirs(os.path.join(root_folder_datasets, 'imagenetv2-thr'), exist_ok=True)
url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz'
filename = wget.download(url, out=os.path.join('datasets', 'imagenetv2-thr', 'imagenetv2-thr.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenetv2-thr', 'imagenetv2-thr.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting ImageNet-V2-THR', total=len(tar.getmembers())):
        # Extract member
        member.name = member.name.replace('imagenetv2-threshold0.7-format-val/', '')
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenetv2-thr'), member=member)
        
os.makedirs(os.path.join(root_folder_datasets, 'imagenetv2-ti'), exist_ok=True)
url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz'
filename = wget.download(url, out=os.path.join('datasets', 'imagenetv2-ti', 'imagenetv2-ti.tar'))
with tarfile.open(name=os.path.join(root_folder_datasets, 'imagenetv2-ti', 'imagenetv2-ti.tar')) as tar:
    # Go over each member
    for member in tqdm.tqdm(iterable=tar.getmembers(), desc='Extracting ImageNet-V2-TI', total=len(tar.getmembers())):
        # Extract member
        member.name = member.name.replace('imagenetv2-top-images-format-val/', '')
        tar.extract(path=os.path.join(root_folder_datasets, 'imagenetv2-ti'), member=member)
