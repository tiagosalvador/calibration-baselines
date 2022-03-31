from main import calibrate

datasets = ['cifar10', 'cifar100', 'svhn']

methods = ['Vanilla',
           'TemperatureScaling',
           'VectorScaling',
           'MatrixScaling',
           'MatrixScalingODIR',
           'DirichletL2',
           'DirichletODIR',
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

calibrate(datasets, architectures, methods, splitIDs)