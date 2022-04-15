from main import evaluate_corrupted

methods = [
    'Vanilla',
    'TemperatureScaling',
    'VectorScaling',
#     'MatrixScaling',
#     'MatrixScalingODIR',
#     'DirichletL2',
#     'DirichletODIR',
]

architectures = [
    'densenet40_k12',
    'resnet20',
    'resnet56',
    'resnet110',
    'wrn16_10',
    'wrn28_10',
    'wrn40_8'
]

splitIDs = [0,1,2,3,4]

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur',
 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
intensities = [1,2,3,4,5]

evaluate_corrupted('cifar100', corruptions, intensities, architectures, methods, splitIDs)