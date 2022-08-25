from main import calibrate, evaluate_ood, evaluate_corrupted, missing_outputs, missing_corrupted


import argparse

parser = argparse.ArgumentParser(description='Benchmarking Post-Hoc Calibration Methods')
parser.add_argument('--splitID', type=int, default=0, help="Split ID")

args = parser.parse_args()

datasets = [
#     'cifar10', 
    'cifar100',
#     'svhn'
#     'imagenet',
]

methods = [
#     'CCAC',
    'CCAC-S',
#     'Vanilla',
#     'TemperatureScaling',
#     'TemperatureScalingMSE',
#     'VectorScaling',
#     'MatrixScaling',
#     'MatrixScalingODIR',
#     'DirichletL2',
#     'DirichletODIR',
#     'EnsembleTemperatureScaling',
#     'EnsembleTemperatureScalingCE',
#     'IRM',
#     'IRMTS',
#     'IROvA',
#     'IROvATS',
#     'TemperatureScaling-P',
#     'TemperatureScalingMSE-P',
#     'EnsembleTemperatureScaling-P',
#     'EnsembleTemperatureScalingCE-P',
#     'IRM-P',
#     'IRMTS-P',
#     'IROvA-P',
#     'IROvATS-P',
]

architectures = {}
architectures['cifar10'] = ['densenet40_k12', 'resnet20', 'resnet56', 'resnet110', 'wrn16_10', 'wrn28_10', 'wrn40_8']
architectures['cifar100'] = ['densenet40_k12', 'resnet20', 'resnet56', 'resnet110', 'wrn16_10', 'wrn28_10', 'wrn40_8']
architectures['cifar100'] = ['wrn40_8', 'wrn28_10', 'wrn16_10', 'resnet110', 'resnet56', 'resnet20', 'densenet40_k12' ]
architectures['svhn'] = ['densenet40_k12', 'resnet20', 'resnet56', 'resnet110', 'wrn16_10', 'wrn28_10', 'wrn40_8']
architectures['imagenet'] = ['resnet50', 'vgg19', 'resnext101_32x8d', 'densenet161', 'wide_resnet101_2']
# splitIDs = [args.splitID]
splitIDs = [0,1,2,3,4]

# missing_outputs(datasets, architectures, methods, splitIDs)
calibrate(datasets, architectures, methods, splitIDs)
# ood_datasets = ['stl10', 'cifar10.1-v4', 'cifar10.1-v6']
# evaluate_ood('cifar10', ood_datasets, architectures['cifar10'], methods, splitIDs)

# ood_datasets = ['imagenet-v2-mf', 'imagenet-v2-thr', 'imagenet-v2-ti', 'imagenet-sketch', 'imagenet-a', 'imagenet-r']
# evaluate_ood('imagenet', ood_datasets, architectures['imagenet'], methods, splitIDs)


corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur',
 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]
corruptions.sort(reverse=True)
intensities = [1,2,3,4,5]

for dataset in ['cifar100']:
    evaluate_corrupted(dataset, corruptions, intensities, architectures[dataset], methods, splitIDs)
# #     missing_corrupted(dataset, corruptions, intensities, architectures[dataset], methods, splitIDs)

