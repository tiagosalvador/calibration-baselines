from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from utils.imagenetv2 import ImageNetv2

import numpy as np
import os

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
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

class CIFAR101(Dataset):
    """CIFAR10.1 dataset."""

    def __init__(self, root, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if 'v4' in root:
            images_file = 'cifar10.1_v4_data.npy'
            labels_file = 'cifar10.1_v4_labels.npy'
        else:
            images_file = 'cifar10.1_v6_data.npy'
            labels_file = 'cifar10.1_v6_labels.npy'
        self.data = np.load(os.path.join(root,images_file))
        self.targets = np.load(os.path.join(root,labels_file))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class CIFARC(Dataset):
    def __init__(self, root, corruption, intensity, transform=None, target_transform=None):
        data = np.load(os.path.join(root,f'{corruption}.npy'))
        data = data[(10000*(intensity-1)):(10000*intensity),:,:,:]
        self.data =  data
        targets = np.load(os.path.join(root,'labels.npy'))[(10000*(intensity-1)):(10000*intensity)]
        self.targets =  torch.LongTensor(targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):                       
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_features_logits_labels(net, subset, ds_info, save=True):
    if subset == 'train':
        suffix_name = 'train.npy'
    else:
        suffix_name = 'test.npy'
    if os.path.exists(os.path.join(ds_info['folder_outputs'], f'labels_{suffix_name}')):
        features = torch.from_numpy(np.load(os.path.join(ds_info['folder_outputs'], f'features_{suffix_name}')))
        logits = torch.from_numpy(np.load(os.path.join(ds_info['folder_outputs'], f'logits_{suffix_name}')))
        labels = torch.from_numpy(np.load(os.path.join(ds_info['folder_outputs'], f'labels_{suffix_name}')))
    else:
        dataloader = get_loader(subset, ds_info)
        net.eval()
        features = []
        logits = []
        with torch.no_grad():
            for data in tqdm(dataloader, desc=f"Generating features, logits and labels", leave=False):
                images, labels = data
                # compute features and logits
                features_temp, logits_temp = net(images.cuda())
                features.append(features_temp.cpu())
                logits.append(logits_temp.cpu())
        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.from_numpy(dataloader.dataset.targets).long()
        if save:
            np.save(os.path.join(ds_info['folder_outputs'], f'features_{suffix_name}'), features)
            np.save(os.path.join(ds_info['folder_outputs'], f'logits_{suffix_name}'), logits)
            np.save(os.path.join(ds_info['folder_outputs'], f'labels_{suffix_name}'), labels)   
    
    if (subset != 'train') and (subset != 'ood'):
        indices = ds_info['indices_'+subset]
        return features[indices], logits[indices], labels[indices]
    else:
        return features, logits, labels

                        
def get_loader(subset, ds_info, shuffle=False):
    if ds_info['name'] == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=ds_info['root_folder_datasets'], train=subset=='train',
                                               download=False, transform=ds_info['transform']);
        dataset.targets = np.array(dataset.targets)
    elif (ds_info['name'] == 'cifar10-c') or (ds_info['name'] == 'cifar100-c'):
        dataset = CIFARC(ds_info['root_folder_datasets'], ds_info['corruption'], ds_info['intensity'], 
                           transform=ds_info['transform']);
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=ds_info['root_folder_datasets'], train=subset=='train',
                                               download=False, transform=ds_info['transform']);
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'svhn':
        split = 'test' if (subset == 'cal' or subset == 'ood') else subset
        dataset = torchvision.datasets.SVHN(root=ds_info['root_folder_datasets'], split=split,
                                               download=False, transform=ds_info['transform']);
        dataset.targets = np.array(dataset.labels)
    elif ds_info['name'] == 'stl10':
        split = 'test' if (subset == 'cal' or subset == 'ood') else subset
        dataset = torchvision.datasets.STL10(root=ds_info['root_folder_datasets'], split=split,
                                               download=False, transform=ds_info['transform']);
        dataset.targets = np.array(dataset.labels)
    elif ds_info['name'] == 'cifar10.1-v4':
        dataset = CIFAR101(root=ds_info['root_folder_datasets'], transform=ds_info['transform']);
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'cifar10.1-v6':
        dataset = CIFAR101(root=ds_info['root_folder_datasets'], transform=ds_info['transform']);
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'imagenet':
        if subset == 'train':
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
        elif subset == 'train_large':
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
        else:
            dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'val')
        dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'imagenet-c':
        dataset_dir = os.path.join(ds_info['root_folder_datasets'], ds_info['corruption'], str(ds_info['intensity']))
        dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        dataset.targets = np.array(dataset.targets)
    elif (ds_info['name'] == 'imagenet-v2-mf') or (ds_info['name'] == 'imagenet-v2-thr') or (ds_info['name'] == 'imagenet-v2-ti'):
        dataset = ImageNetv2(ds_info['root_folder_datasets'], transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'imagenet-sketch':
        dataset = torchvision.datasets.ImageFolder(ds_info['root_folder_datasets'], transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        dataset.targets = np.array(dataset.targets)
    elif ds_info['name'] == 'imagenet-a':
        dataset = torchvision.datasets.ImageFolder(ds_info['root_folder_datasets'], transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        map_indices = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207,234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988]
        dataset.targets = np.array([map_indices[idx] for idx in np.array(dataset.targets)])
    elif ds_info['name'] == 'imagenet-r':
        dataset = torchvision.datasets.ImageFolder(ds_info['root_folder_datasets'], transform=ds_info['transform'])
        dataset.samples = np.array(dataset.samples)
        map_indices = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]
        dataset.targets = np.array([map_indices[idx] for idx in np.array(dataset.targets)])
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds_info['batch_size'], shuffle=shuffle, num_workers=4)
    return loader


# def get_features_logits_labels_old(net, subset, ds_info, save=True):
#     experiments_folder = ds_info['folder']
#     suffix_name = '.npy'
#     folder_path = os.path.join(experiments_folder,'features_logits_labels')
#     features = None
#     if os.path.exists(os.path.join(folder_path, 'labels_'+subset+suffix_name)):
# #         features = torch.from_numpy(np.load(os.path.join(folder_path, 'features_'+subset+suffix_name)))
#         logits = torch.from_numpy(np.load(os.path.join(folder_path, 'logits_'+subset+suffix_name)))
#         labels = torch.from_numpy(np.load(os.path.join(folder_path, 'labels_'+subset+suffix_name)))
#     else:
#         dataloader = get_loader(subset, ds_info)
#         net.eval()
# #         features = torch.zeros(len(dataloader.dataset.targets), ds_info['dim_features'], dtype=float)
#         logits = []
#         start = 0
#         end = 0
#         with torch.no_grad():
#             for data in tqdm(dataloader, desc=f"Generating features, logits and labels", leave=False):
#                 images, labels = data
#                 end += len(images)
#                 # compute features and logits
# #                 features_temp = net.feature_extractor(images.cuda())
# #                 logits.append(net.output(features_temp).cpu())
#                 logits.append(net(images.cuda()).cpu())
# #                 if 'squeezenet' in ds_info['architecture']:
# #                     features[start:end] = features_temp.cpu().reshape(features_temp.shape[0],-1)
# #                 else:
# #                     features[start:end] = features_temp.cpu()            
#                 start = end
#         logits = torch.cat(logits,dim=0)
#         labels = torch.from_numpy(dataloader.dataset.targets).long()
#         if save:
# #             np.save(os.path.join(folder_path, 'features_'+subset+suffix_name), features)
#             np.save(os.path.join(folder_path, 'logits_'+subset+suffix_name), logits)
#             np.save(os.path.join(folder_path, 'labels_'+subset+suffix_name), labels)    

#     return features, logits, labels

# def get_loader_old(subset, ds_info, shuffle=False):
#     if (subset != 'train') and (subset != 'ood'):
#         indices = ds_info['indices_'+subset]
#     if ds_info['name'] == 'cifar10':
#         dataset = torchvision.datasets.CIFAR10(root=ds_info['root_folder_datasets'], train=subset=='train',
#                                                download=False, transform=ds_info['transform']);
#         if (subset != 'train') and (subset != 'ood'):
#             dataset.data = dataset.data[indices]
#             dataset.targets = np.array(dataset.targets)[indices]
#         else:
#             dataset.targets = np.array(dataset.targets)
#     elif (ds_info['name'] == 'cifar10-c') or (ds_info['name'] == 'cifar100-c'):
#         dataset = CIFARC(ds_info['root_folder_datasets'], ds_info['corruption'], ds_info['intensity'], 
#                            transform=ds_info['transform']);
#         if (subset != 'train') and (subset != 'ood'):
#             dataset.data = dataset.data[indices]
#             dataset.targets = np.array(dataset.targets)[indices]
#         else:
#             dataset.targets = np.array(dataset.targets)
#     elif ds_info['name'] == 'cifar100':
#         dataset = torchvision.datasets.CIFAR100(root=ds_info['root_folder_datasets'], train=subset=='train',
#                                                download=False, transform=ds_info['transform']);
#         if (subset != 'train') and (subset != 'ood'):
#             dataset.data = dataset.data[indices]
#             dataset.targets = np.array(dataset.targets)[indices]
#         else:
#             dataset.targets = np.array(dataset.targets)
#     elif ds_info['name'] == 'svhn':
#         split = 'test' if (subset == 'cal' or subset == 'ood') else subset
#         dataset = torchvision.datasets.SVHN(root=ds_info['root_folder_datasets'], split=split,
#                                                download=False, transform=ds_info['transform']);
#         if (subset != 'train') and (subset != 'ood'):
#             dataset.data = dataset.data[indices]
#             dataset.targets = np.array(dataset.labels)[indices]
#         else:
#             dataset.targets = np.array(dataset.labels)
#     elif ds_info['name'] == 'stl10':
#         split = 'test' if (subset == 'cal' or subset == 'ood') else subset
#         dataset = torchvision.datasets.STL10(root=ds_info['root_folder_datasets'], split=split,
#                                                download=False, transform=ds_info['transform']);
#         if (subset != 'train') and (subset != 'ood'):
#             dataset.data = dataset.data[indices]
#             dataset.targets = np.array(dataset.labels)[indices]
#         else:
#             dataset.targets = np.array(dataset.labels)
#     elif ds_info['name'] == 'cifar10.1-v4':
#         dataset = CIFAR101(root=ds_info['root_folder_datasets'], transform=ds_info['transform']);
#         dataset.targets = np.array(dataset.targets)
#     elif ds_info['name'] == 'cifar10.1-v6':
#         dataset = CIFAR101(root=ds_info['root_folder_datasets'], transform=ds_info['transform']);
#         dataset.targets = np.array(dataset.targets)
#     elif ds_info['name'] == 'imagenet':
#         if subset == 'train':
#             dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
#         elif subset == 'train_large':
#             dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'train')
#         else:
#             dataset_dir = os.path.join(ds_info['root_folder_datasets'], 'val')
#         dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
#         dataset.samples = np.array(dataset.samples)[indices]
#         dataset.targets = np.array(dataset.targets)[indices]
#     elif ds_info['name'] == 'imagenet-c':
#         dataset_dir = os.path.join(ds_info['root_folder_datasets'], ds_info['corruption'], str(ds_info['intensity']))
#         dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=ds_info['transform'])
#         dataset.samples = np.array(dataset.samples)[indices]
#         dataset.targets = np.array(dataset.targets)[indices]        
#     loader = torch.utils.data.DataLoader(dataset, batch_size=ds_info['batch_size'], shuffle=shuffle, num_workers=4)
#     return loader
