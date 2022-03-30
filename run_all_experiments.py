from main import calibrate

datasets = ['cifar10']

methods = ['Vanilla',
           'TemperatureScaling',
#            'VectorScaling',
#            'MatrixScaling',
#            'MatrixScalingODIR',
#            'DirichletL2',
           'DirichletODIR',
          ]
architectures = [
    'resnet20',
    'resnet56',
#     'resnet110',
#     'resnet164bn',
#     'resnet272bn',
#     'resnet542bn',
#     'resnet1001',
#     'resnet1202'
]
splitIDs = [0,1,2,3,4]

calibrate(datasets, architectures, methods, splitIDs)