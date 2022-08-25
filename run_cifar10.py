from main import calibrate, evaluate_ood, evaluate_corrupted

datasets = ['cifar10']

methods = [
    'Vanilla',
    'TemperatureScaling',
    'VectorScaling',
    'MatrixScaling',
    'MatrixScalingODIR',
    'DirichletL2',
    'DirichletODIR',
    'EnsembleTemperatureScaling',
    'IRM',
    'IRMTS',
    'IROvA',
    'IROvATS',
]

architectures = {}
architectures['cifar10'] = ['densenet40_k12', 'resnet20', 'resnet56', 'wrn16_10', 'wrn28_10']
splitIDs = [0,1,2,3,4]

calibrate(datasets, architectures, methods, splitIDs)
ood_datasets = ['stl10', 'cifar10.1-v4', 'cifar10.1-v6']
evaluate_ood('cifar10', ood_datasets, architectures['cifar10'], methods, splitIDs)

corruptions = [
 'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur',
 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]
corruptions.sort(reverse=True)
intensities = [1,2,3,4,5]

for dataset in ['cifar10']:
    evaluate_corrupted(dataset, corruptions, intensities, architectures[dataset], methods, splitIDs)
