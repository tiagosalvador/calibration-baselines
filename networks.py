import torch
import torchvision
from torch import nn

from pytorchcv.model_provider import get_model as ptcv_get_model
from imgclsmob.pytorch.utils import prepare_model as prepare_model_pt

class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model, architecture):
        super(FeatureExtractorWrapper, self).__init__()
        self.architecture = architecture
        if 'alexnet' in architecture or 'vgg' in architecture:
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
            self.dim_features = self.classifier[-1].in_features
        elif 'squeezenet' in architecture:
            self.features = model.features
            self.classifier = model.classifier
            self.dim_features = 512*13*13
        elif 'densenet' in architecture:
            self.features = model.features
            self.classifier = model.classifier
            self.dim_features = self.classifier.in_features
        elif 'inception_v3' == architecture or 'googlenet' == architecture:
            self._transform_input = model._transform_input
            self.features = torch.nn.Sequential(*list(model.children())[:-1])
            self.fc = model.fc
            self.dim_features = self.fc.in_features
        elif 'shufflenet' in architecture:
            self.features = torch.nn.Sequential(*list(model.children())[:-1])
            self.fc = model.fc
            self.dim_features = self.fc.in_features
        elif 'mobilenet' in architecture:
            self.features = model.features
            self.classifier = model.classifier
            self.dim_features = self.classifier[-1].in_features
        elif 'resnet' in architecture or 'resnext' in architecture:
            self.features = torch.nn.Sequential(*list(model.children())[:-1])
            self.fc = model.fc
            self.dim_features = self.fc.in_features
        elif 'mnasnet' in architecture:
            self.layers = model.layers
            self.classifier = model.classifier
            self.dim_features = self.classifier[-1].in_features
    def feature_extractor(self, x):
        if 'alexnet' == self.architecture or 'vgg' in self.architecture:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier[:-1](x)
        elif 'squeezenet' in self.architecture:
            x = self.features(x)
        elif 'densenet' in self.architecture:
            x = self.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif 'inception_v3' == self.architecture or 'googlenet' == self.architecture:
            x = self._transform_input(x)
            x = self.features(x)
            x = torch.flatten(x,1)
        elif 'shufflenet' in self.architecture:
            x = self.features(x)
            x = x.mean([2, 3])
        elif 'mobilenet' in self.architecture:
            x = self.features(x)
            if 'v2' in self.architecture:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
            elif 'v3' in self.architecture:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
        elif 'resnet' in self.architecture or 'resnext' in self.architecture:
            x = self.features(x)
            x = torch.flatten(x,1)
        elif 'mnasnet' in self.architecture:
            x = self.layers(x)
            x = x.mean([2,3])
        return x
    def output(self, x):
        if 'alexnet' == self.architecture or 'vgg' in self.architecture:
            x = self.classifier[-1](x)
        elif 'squeezenet' in self.architecture:
            x = self.classifier(x)
            x = torch.flatten(x, 1)
        elif 'densenet' in self.architecture:
            x = self.classifier(x)
        elif 'inception_v3' == self.architecture or 'googlenet' == self.architecture:
            x = self.fc(x)
        elif 'shufflenet' in self.architecture:
            x = self.fc(x)
        elif 'mobilenet' in self.architecture:
            x = self.classifier(x)
        elif ('resnet' in self.architecture) or ('resnext' in self.architecture):
            x = self.fc(x)
        elif 'mnasnet' in self.architecture:
            x = self.classifier(x)
        return x
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.output(features)
        return features, logits
    
class FeatureExtractorWrapperCIFAR10(nn.Module):
    def __init__(self, model, architecture):
        super(FeatureExtractorWrapperCIFAR10, self).__init__()
        self.features = model.features
        self.output = model.output
        self.architecture = architecture
        self.dim_features = model.output.in_features
        self.parallel = False
    
    def feature_extractor(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self,x):
        features = self.feature_extractor(x)
        logits = self.output(features)
        return features, logits

def load_net(dataset, architecture, device):
    if (dataset == 'cifar10') or (dataset == 'cifar100') or (dataset == 'svhn'):
        net = ptcv_get_model(f'{architecture}_{dataset}', pretrained=True)
        net = FeatureExtractorWrapperCIFAR10(net, architecture)
    elif dataset == 'imagenet':
        if architecture == 'alexnet':
            net_ft = torchvision.models.alexnet(pretrained=True)
        elif architecture == 'vgg11':
            net_ft = torchvision.models.vgg11(pretrained=True)
        elif architecture == 'vgg11_bn':
            net_ft = torchvision.models.vgg11_bn(pretrained=True)
        elif architecture == 'vgg13':
            net_ft = torchvision.models.vgg13(pretrained=True)
        elif architecture == 'vgg13_bn':
            net_ft = torchvision.models.vgg13_bn(pretrained=True)
        elif architecture == 'vgg16':
            net_ft = torchvision.models.vgg16(pretrained=True)
        elif architecture == 'vgg16_bn':
            net_ft = torchvision.models.vgg16_bn(pretrained=True)
        elif architecture == 'vgg19':
            net_ft = torchvision.models.vgg19(pretrained=True)
        elif architecture == 'vgg19_bn':
            net_ft = torchvision.models.vgg19_bn(pretrained=True)
        elif architecture == 'resnet18':
            net_ft = torchvision.models.resnet18(pretrained=True)
        elif architecture == 'resnet34':
            net_ft = torchvision.models.resnet34(pretrained=True)
        elif architecture == 'resnet50':
            net_ft = torchvision.models.resnet50(pretrained=True)
        elif architecture == 'resnet101':
            net_ft = torchvision.models.resnet101(pretrained=True)        
        elif architecture == 'resnet152':
            net_ft = torchvision.models.resnet152(pretrained=True)
        elif architecture == 'squeezenet1_0':
            net_ft = torchvision.models.squeezenet1_0(pretrained=True)
        elif architecture == 'squeezenet1_1':
            net_ft = torchvision.models.squeezenet1_1(pretrained=True)
        elif architecture == 'densenet121':
            net_ft = torchvision.models.densenet121(pretrained=True)
        elif architecture == 'densenet169':
            net_ft = torchvision.models.densenet169(pretrained=True)
        elif architecture == 'densenet161':
            net_ft = torchvision.models.densenet161(pretrained=True)
        elif architecture == 'densenet201':
            net_ft = torchvision.models.densenet201(pretrained=True)
        elif architecture == 'inception_v3':
            net_ft = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        elif architecture == 'googlenet':
            net_ft = torchvision.models.googlenet(pretrained=True)
        elif architecture == 'shufflenet_v2_x0_5':
            net_ft = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        elif architecture == 'shufflenet_v2_x1_0':
            net_ft = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        elif architecture == 'shufflenet_v2_x1_5':
            net_ft = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
        elif architecture == 'shufflenet_v2_x2_0':
            net_ft = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
        elif architecture == 'mobilenet_v2':
            net_ft = torchvision.models.mobilenet_v2(pretrained=True)
        elif architecture == 'mobilenet_v3_large':
            net_ft = torchvision.models.mobilenet_v3_large(pretrained=True)
        elif architecture == 'mobilenet_v3_small':
            net_ft = torchvision.models.mobilenet_v3_small(pretrained=True)
        elif architecture == 'resnext50_32x4d':
            net_ft = torchvision.models.resnext50_32x4d(pretrained=True)
        elif architecture == 'resnext101_32x8d':
            net_ft = torchvision.models.resnext101_32x8d(pretrained=True)
        elif architecture == 'wide_resnet50_2':
            net_ft = torchvision.models.wide_resnet50_2(pretrained=True)
        elif architecture == 'wide_resnet101_2':
            net_ft = torchvision.models.wide_resnet101_2(pretrained=True)
        elif architecture == 'mnasnet0_5':
            net_ft = torchvision.models.mnasnet0_5(pretrained=True)
        elif architecture == 'mnasnet0_75':
            net_ft = torchvision.models.mnasnet0_75(pretrained=True)
        elif architecture == 'mnasnet1_0':
            net_ft = torchvision.models.mnasnet1_0(pretrained=True)
        elif architecture == 'mnasnet1_3':
            net_ft = torchvision.models.mnasnet1_3(pretrained=True)
#         elif architecture == 'efficientnet_b0':
#             net_ft = torchvision.models.efficientnet_b0(pretrained=True)
#         elif architecture == 'efficientnet_b1':
#             net_ft = torchvision.models.efficientnet_b1(pretrained=True)
#         elif architecture == 'efficientnet_b2':
#             net_ft = torchvision.models.efficientnet_b2(pretrained=True)
#         elif architecture == 'efficientnet_b3':
#             net_ft = torchvision.models.efficientnet_b3(pretrained=True)
#         elif architecture == 'efficientnet_b4':
#             net_ft = torchvision.models.efficientnet_b4(pretrained=True)
#         elif architecture == 'efficientnet_b5':
#             net_ft = torchvision.models.efficientnet_b5(pretrained=True)
#         elif architecture == 'efficientnet_b6':
#             net_ft = torchvision.models.efficientnet_b6(pretrained=True)
#         elif architecture == 'efficientnet_b7':
#             net_ft = torchvision.models.efficientnet_b7(pretrained=True)
#         elif architecture == 'regnet_y_400mf':
#             net_ft = torchvision.models.regnet_y_400mf(pretrained=True)
#         elif architecture == 'regnet_y_800mf':
#             net_ft = torchvision.models.regnet_y_800mf(pretrained=True)
#         elif architecture == 'regnet_y_1_6gf':
#             net_ft = torchvision.models.regnet_y_1_6gf(pretrained=True)
#         elif architecture == 'regnet_y_3_2gf':
#             net_ft = torchvision.models.regnet_y_3_2gf(pretrained=True)
#         elif architecture == 'regnet_y_8gf':
#             net_ft = torchvision.models.regnet_y_8gf(pretrained=True)
#         elif architecture == 'regnet_y_16gf':
#             net_ft = torchvision.models.regnet_y_16gf(pretrained=True)
#         elif architecture == 'regnet_y_32gf':
#             net_ft = torchvision.models.regnet_y_32gf(pretrained=True)
#         elif architecture == 'regnet_x_400mf':
#             net_ft = torchvision.models.regnet_x_400mf(pretrained=True)
#         elif architecture == 'regnet_x_800mf':
#             net_ft = torchvision.models.regnet_x_800mf(pretrained=True)
#         elif architecture == 'regnet_x_1_6gf':
#             net_ft = torchvision.models.regnet_x_1_6gf(pretrained=True)
#         elif architecture == 'regnet_x_3_2gf':
#             net_ft = torchvision.models.regnet_x_3_2gf(pretrained=True)
#         elif architecture == 'regnet_x_8gf':
#             net_ft = torchvision.models.regnet_x_8gf(pretrained=True)
#         elif architecture == 'regnet_x_16gf':
#             net_ft = torchvision.models.regnet_x_16gf(pretrained=True)
#         elif architecture == 'regnet_x_32gf':
#             net_ft = torchvision.models.regnet_x_32gf(pretrained=True)
        else:
            raise ValueError("Unsupported model: {}".format(architecture))
        net = FeatureExtractorWrapper(net_ft, architecture)
        if 'cuda' in device.type:
            net = nn.DataParallel(net, device_ids=[0,1])
            net.architecture = net.module.architecture
            net.dim_features = net.module.dim_features
            net.parallel = True

    net.to(device);
    return net