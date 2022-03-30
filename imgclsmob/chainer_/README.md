# Computer vision models on Chainer

[![PyPI](https://img.shields.io/pypi/v/chainercv2.svg)](https://pypi.python.org/pypi/chainercv2)
[![Downloads](https://pepy.tech/badge/chainercv2)](https://pepy.tech/project/chainercv2)

This is a collection of image classification, segmentation, detection, and pose estimation models. Many of them are pretrained on
[ImageNet-1K](http://www.image-net.org), [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html),
[SVHN](http://ufldl.stanford.edu/housenumbers), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012), [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K),
[Cityscapes](https://www.cityscapes-dataset.com), and [COCO](http://cocodataset.org) datasets and loaded automatically
during use. All pretrained models require the same ordinary normalization. Scripts for training/evaluating/converting
models are in the [`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- BN-Inception (['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- ResNeSt(A) (['ResNeSt: Split-Attention Networks'](https://arxiv.org/abs/2004.08955))
- AirNet/AirNeXt (['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896))
- BAM-ResNet (['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514))
- CBAM-ResNet (['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521))
- ResAttNet (['Residual Attention Network for Image Classification'](https://arxiv.org/abs/1704.06904))
- SKNet (['Selective Kernel Networks'](https://arxiv.org/abs/1903.06586))
- SCNet (['Improving Convolutional Networks with Self-Calibrated Convolutions'](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf))
- RegNet (['Designing Network Design Spaces'](https://arxiv.org/abs/2003.13678))
- DIA-ResNet (['DIANet: Dense-and-Implicit Attention Network'](https://arxiv.org/abs/1905.10671))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- ShaResNet (['ShaResNet: reducing residual network parameter number by sharing weights'](https://arxiv.org/abs/1702.08782))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- SparseNet (['Sparsely Aggregated Convolutional Networks'](https://arxiv.org/abs/1801.05895))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- Oct-ResNet (['Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution'](https://arxiv.org/abs/1904.05049))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- WRN-1bit (['Training wide residual networks for deployment using a single bit for each weight'](https://arxiv.org/abs/1802.08530))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- ChannelNet (['ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'](https://arxiv.org/abs/1809.01330))
- i-RevNet (['i-RevNet: Deep Invertible Networks'](https://arxiv.org/abs/1802.07088))
- BagNet (['Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet'](https://openreview.net/pdf?id=SkfMWhAqYQ))
- DLA (['Deep Layer Aggregation'](https://arxiv.org/abs/1707.06484))
- FishNet (['FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction'](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf))
- ESPNetv2 (['ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network'](https://arxiv.org/abs/1811.11431))
- DiCENet (['DiCENet: Dimension-wise Convolutions for Efficient Networks'](https://arxiv.org/abs/1906.03516))
- HRNet (['Deep High-Resolution Representation Learning for Visual Recognition'](https://arxiv.org/abs/1908.07919))
- VoVNet (['An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection'](https://arxiv.org/abs/1904.09730))
- SelecSLS (['XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera'](https://arxiv.org/abs/1907.00837))
- HarDNet (['HarDNet: A Low Memory Traffic Network'](https://arxiv.org/abs/1909.00948))
- X-DenseNet (['Deep Expander Networks: Efficient Deep Networks from Graph Theory'](https://arxiv.org/abs/1711.08757))
- SqueezeNet/SqueezeResNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- MobileNetV3 (['Searching for MobileNetV3'](https://arxiv.org/abs/1905.02244))
- IGCV3 (['IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks'](https://arxiv.org/abs/1806.00178))
- GhostNet (['GhostNet: More Features from Cheap Operations'](https://arxiv.org/abs/1911.11907))
- MnasNet (['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626))
- DARTS (['DARTS: Differentiable Architecture Search'](https://arxiv.org/abs/1806.09055))
- ProxylessNAS (['ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware'](https://arxiv.org/abs/1812.00332))
- FBNet (['FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search'](https://arxiv.org/abs/1812.03443))
- Xception (['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357))
- InceptionV3 (['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567))
- InceptionV4/InceptionResNetV1/InceptionResNetV2 (['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261))
- PolyNet (['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725))
- NASNet (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))
- PNASNet (['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559))
- SPNASNet (['Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours'](https://arxiv.org/abs/1904.02877))
- EfficientNet (['EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'](https://arxiv.org/abs/1905.11946))
- MixNet (['MixConv: Mixed Depthwise Convolutional Kernels'](https://arxiv.org/abs/1907.09595))
- NIN (['Network In Network'](https://arxiv.org/abs/1312.4400))
- RoR-3 (['Residual Networks of Residual Networks: Multilevel Residual Networks'](https://arxiv.org/abs/1608.02908))
- RiR (['Resnet in Resnet: Generalizing Residual Architectures'](https://arxiv.org/abs/1603.08029))
- ResDrop-ResNet (['Deep Networks with Stochastic Depth'](https://arxiv.org/abs/1603.09382))
- Shake-Shake-ResNet (['Shake-Shake regularization'](https://arxiv.org/abs/1705.07485))
- ShakeDrop-ResNet (['ShakeDrop Regularization for Deep Residual Learning'](https://arxiv.org/abs/1802.02375))
- NTS-Net (['Learning to Navigate for Fine-grained Classification'](https://arxiv.org/abs/1809.00287))
- PSPNet (['Pyramid Scene Parsing Network'](https://arxiv.org/abs/1612.01105))
- DeepLabv3 (['Rethinking Atrous Convolution for Semantic Image Segmentation'](https://arxiv.org/abs/1706.05587))
- FCN-8s (['Fully Convolutional Networks for Semantic Segmentation'](https://arxiv.org/abs/1411.4038))
- ICNet (['ICNet for Real-Time Semantic Segmentation on High-Resolution Images'](https://arxiv.org/abs/1704.08545))
- Fast-SCNN (['Fast-SCNN: Fast Semantic Segmentation Network'](https://arxiv.org/abs/1902.04502))
- CGNet (['CGNet: A Light-weight Context Guided Network for Semantic Segmentation'](https://arxiv.org/abs/1811.08201))
- DABNet (['DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation'](https://arxiv.org/abs/1907.11357))
- SINet (['SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder'](https://arxiv.org/abs/1911.09099))
- BiSeNet (['BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation'](https://arxiv.org/abs/1808.00897))
- DANet (['Dual Attention Network for Scene Segmentation'](https://arxiv.org/abs/1809.02983))
- FPENet (['Feature Pyramid Encoding Network for Real-time Semantic Segmentation'](https://arxiv.org/abs/1909.08599))
- LEDNet (['LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation'](https://arxiv.org/abs/1905.02423))
- CenterNet (['Objects as Points'](https://arxiv.org/abs/1904.07850))
- LFFD (['LFFD: A Light and Fast Face Detector for Edge Devices'](https://arxiv.org/abs/1904.10633))
- AlphaPose (['RMPE: Regional Multi-person Pose Estimation'](https://arxiv.org/abs/1612.00137))
- SimplePose (['Simple Baselines for Human Pose Estimation and Tracking'](https://arxiv.org/abs/1804.06208))
- Lightweight OpenPose (['Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose'](https://arxiv.org/abs/1811.12004))
- IBPPose (['Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation'](https://arxiv.org/abs/1911.10529))
- VOCA (['Capture, Learning, and Synthesis of 3D Speaking Styles'](https://arxiv.org/abs/1905.03079))
- Neural Voice Puppetry Audio-to-Expression net (['Neural Voice Puppetry: Audio-driven Facial Reenactment'](https://arxiv.org/abs/1912.05566))
- Jasper/JasperDR (['Jasper: An End-to-End Convolutional Neural Acoustic Model'](https://arxiv.org/abs/1904.03288))
- QuartzNet (['QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions'](https://arxiv.org/abs/1910.10261))

## Installation

To use the models in your project, simply install the `chainercv2` package:
```
pip install chainercv2
```
To enable/disable different hardware supports, check out Chainer installation [instructions](https://chainer.org).

## Usage

Example of using a pretrained ResNet-18 model:
```
from chainercv2.model_provider import get_model as chcv2_get_model
import numpy as np

net = chcv2_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net(x)
```

## Pretrained models

### ImageNet-1K

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to Chainer.
- ResNet(A) is an average downsampled ResNet intended for use as an feature extractor in some pose estimation networks.
- ResNet(D) is a dilated ResNet intended for use as an feature extractor in some segmentation networks.
- Models with *-suffix use non-standard preprocessing (see the training log).

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 38.04 | 16.10 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/alexnet-1610-d666015b.npz.log)) |
| AlexNet-b | 39.27 | 17.05 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/alexnetb-1705-a22a3ab8.npz.log)) |
| ZFNet | 39.19 | 16.75 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.395/zfnet-1675-0205a9ab.npz.log)) |
| ZFNet-b | 35.80 | 14.56 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.400/zfnetb-1456-5808c73e.npz.log)) |
| VGG-11 | 29.60 | 10.17 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.381/vgg11-1017-7934dcf0.npz.log)) |
| VGG-13 | 28.47 | 9.52 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.388/vgg13-0952-f6af5a26.npz.log)) |
| VGG-16 | 26.63 | 8.33 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.401/vgg16-0833-5e08a9ec.npz.log)) |
| VGG-19 | 25.59 | 7.66 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.420/vgg19-0766-abf32909.npz.log)) |
| BN-VGG-11 | 28.57 | 9.37 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.339/bn_vgg11-0937-8fcdb341.npz.log)) |
| BN-VGG-13 | 27.67 | 8.87 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.353/bn_vgg13-0887-1709fd1a.npz.log)) |
| BN-VGG-16 | 25.45 | 7.59 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.359/bn_vgg16-0759-8d6a2a82.npz.log)) |
| BN-VGG-19 | 23.89 | 6.88 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.360/bn_vgg19-0688-5b6f413c.npz.log)) |
| BN-VGG-11b | 29.29 | 9.78 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.407/bn_vgg11b-0978-54b2345e.npz.log)) |
| BN-VGG-13b | 28.24 | 9.16 | 133,053,736 | 11,342.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.488/bn_vgg13b-0916-e0110b44.npz.log)) |
| BN-VGG-16b | 25.78 | 7.75 | 138,365,992 | 15,507.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/bn_vgg16b-0775-03703844.npz.log)) |
| BN-VGG-19b | 24.83 | 7.33 | 143,678,248 | 19,672.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/bn_vgg19b-0733-44d38dbe.npz.log)) |
| BN-Inception | 25.11 | 7.52 | 11,295,240 | 2,048.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.405/bninception-0752-44a9e12c.npz.log)) |
| ResNet-10 | 32.58 | 12.55 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/resnet10-1255-bc5960a1.npz.log)) |
| ResNet-12 | 31.62 | 12.04 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/resnet12-1204-651ffc1c.npz.log)) |
| ResNet-14 | 30.36 | 10.93 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/resnet14-1093-adafc1c1.npz.log)) |
| ResNet-BC-14b | 29.22 | 10.36 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/resnetbc14b-1036-8c665d1b.npz.log)) |
| ResNet-16 | 28.56 | 9.78 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnet16-0978-d2b6300f.npz.log)) |
| ResNet-18 x0.25 | 39.32 | 17.45 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.262/resnet18_wd4-1745-79de61de.npz.log)) |
| ResNet-18 x0.5 | 33.41 | 12.85 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.263/resnet18_wd2-1285-ae41e11d.npz.log)) |
| ResNet-18 x0.75 | 29.98 | 10.67 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.266/resnet18_w3d4-1067-4defa49f.npz.log)) |
| ResNet-18 | 26.75 | 8.68 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.478/resnet18-0868-6e670b22.npz.log)) |
| ResNet-26 | 25.98 | 8.24 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnet26-0824-0ae9add4.npz.log)) |
| ResNet-BC-26b | 24.82 | 7.55 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.313/resnetbc26b-0755-74cf9fe9.npz.log)) |
| ResNet-34 | 24.51 | 7.46 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.291/resnet34-0746-1856e049.npz.log)) |
| ResNet-BC-38b | 23.47 | 6.75 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.328/resnetbc38b-0675-9210464e.npz.log)) |
| ResNet-50 | 22.09 | 6.07 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.329/resnet50-0607-f4a16228.npz.log)) |
| ResNet-50b | 22.08 | 6.15 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.308/resnet50b-0615-32bc835e.npz.log)) |
| ResNet-101 | 20.54 | 5.17 | 44,549,160 | 7,597.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/resnet101-0517-88c52266.npz.log)) |
| ResNet-101b | 20.28 | 5.14 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.357/resnet101b-0514-077eb1e2.npz.log)) |
| ResNet-152 | 19.13 | 4.47 | 60,192,808 | 11,321.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.518/resnet152-0447-99b827c4.npz.log)) |
| ResNet-152b | 18.87 | 4.30 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.517/resnet152b-0430-c0473952.npz.log)) |
| PreResNet-10 | 34.70 | 14.02 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.249/preresnet10-1402-94e8fc28.npz.log)) |
| PreResNet-12 | 33.62 | 13.18 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.257/preresnet12-1318-fea1c8c5.npz.log)) |
| PreResNet-14 | 32.28 | 12.24 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.260/preresnet14-1224-f9973f4f.npz.log)) |
| PreResNet-BC-14b | 30.67 | 11.53 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.315/preresnetbc14b-1153-1d37e533.npz.log)) |
| PreResNet-16 | 30.22 | 10.80 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.261/preresnet16-1080-ac7a346a.npz.log)) |
| PreResNet-18 x0.25 | 39.58 | 17.78 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.272/preresnet18_wd4-1778-1cf8aa48.npz.log)) |
| PreResNet-18 x0.5 | 33.64 | 13.12 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.273/preresnet18_wd2-1312-fa4ce56a.npz.log)) |
| PreResNet-18 x0.75 | 29.95 | 10.69 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.274/preresnet18_w3d4-1069-25ddcd56.npz.log)) |
| PreResNet-18 | 28.17 | 9.54 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0954-21e4811a.npz.log)) |
| PreResNet-26 | 25.99 | 8.38 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.316/preresnet26-0838-8cbc7638.npz.log)) |
| PreResNet-BC-26b | 25.20 | 7.86 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.325/preresnetbc26b-0786-4c1e6a24.npz.log)) |
| PreResNet-34 | 24.58 | 7.55 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.300/preresnet34-0755-b664c649.npz.log)) |
| PreResNet-BC-38b | 22.70 | 6.36 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.348/preresnetbc38b-0636-3105fbe8.npz.log)) |
| PreResNet-50 | 22.21 | 6.24 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.330/preresnet50-0624-a2bba5b6.npz.log)) |
| PreResNet-50b | 22.31 | 6.34 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.307/preresnet50b-0634-605b0eec.npz.log)) |
| PreResNet-101 | 20.58 | 5.34 | 44,541,608 | 7,595.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.504/preresnet101-0534-b4e33e60.npz.log)) |
| PreResNet-101b | 20.87 | 5.38 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.351/preresnet101b-0538-b502bf25.npz.log)) |
| PreResNet-152 | 19.16 | 4.47 | 60,185,256 | 11,319.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.510/preresnet152-0447-f62a6bc8.npz.log)) |
| PreResNet-152b | 19.00 | 4.38 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.523/preresnet152b-0438-50121ca1.npz.log)) |
| PreResNet-200b | 18.96 | 4.46 | 64,666,280 | 15,068.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.529/preresnet200b-0446-f44d47e8.npz.log)) |
| PreResNet-269b | 20.17 | 5.04 | 102,065,832 | 20,101.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.545/preresnet269b-0504-c9363468.npz.log)) |
| ResNeXt-14 (16x4d) | 31.63 | 12.26 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.370/resnext14_16x4d-1226-80d9a331.npz.log)) |
| ResNeXt-14 (32x2d) | 32.13 | 12.49 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.371/resnext14_32x2d-1249-892f96a4.npz.log)) |
| ResNeXt-14 (32x4d) | 30.01 | 11.15 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.327/resnext14_32x4d-1115-fa0e7f7f.npz.log)) |
| ResNeXt-26 (32x2d) | 26.32 | 8.49 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.373/resnext26_32x2d-0849-58d86996.npz.log)) |
| ResNeXt-26 (32x4d) | 23.91 | 7.19 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.332/resnext26_32x4d-0719-62ca5090.npz.log)) |
| ResNeXt-50 (32x4d) | 20.84 | 5.46 | 25,028,904 | 4,255.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.498/resnext50_32x4d-0546-d1e20cc5.npz.log)) |
| ResNeXt-101 (32x4d) | 18.51 | 4.16 | 44,177,704 | 8,003.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.530/resnext101_32x4d-0416-ec66f27f.npz.log)) |
| ResNeXt-101 (64x4d) | 18.82 | 4.38 | 83,455,272 | 15,500.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.544/resnext101_64x4d-0438-06bfc635.npz.log)) |
| SE-ResNet-10 | 31.41 | 11.70 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/seresnet10-1170-2b3424cb.npz.log)) |
| SE-ResNet-12 | 31.64 | 11.76 | 5,537,896 | 1,126.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.544/seresnet12-1176-0095f8bd.npz.log)) |
| SE-ResNet-14 | 30.39 | 11.00 | 5,835,504 | 1,358.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.545/seresnet14-1100-fb477b24.npz.log)) |
| SE-ResNet-16 | 28.33 | 9.71 | 7,024,640 | 1,589.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.545/seresnet16-0971-c15d615a.npz.log)) |
| SE-ResNet-18 | 27.96 | 9.23 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.355/seresnet18-0923-b0931abe.npz.log)) |
| SE-ResNet-26 | 25.40 | 8.06 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.363/seresnet26-0806-00032d5b.npz.log)) |
| SE-ResNet-BC-26b | 23.42 | 6.84 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.366/seresnetbc26b-0684-884c0e6b.npz.log)) |
| SE-ResNet-BC-38b | 21.40 | 5.79 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.374/seresnetbc38b-0579-7f103cd0.npz.log)) |
| SE-ResNet-50 | 21.11 | 5.59 | 28,088,024 | 3,883.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.441/seresnet50-0559-6c5585d5.npz.log)) |
| SE-ResNet-50b | 20.56 | 5.30 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.387/seresnet50b-0530-1ac3bf50.npz.log)) |
| SE-ResNet-101 | 19.02 | 4.41 | 49,326,872 | 7,602.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.533/seresnet101-0441-c70e5190.npz.log)) |
| SE-ResNet-101b | 19.48 | 4.63 | 49,326,872 | 7,839.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.460/seresnet101b-0463-97cc55c3.npz.log)) |
| SE-ResNet-152 | 18.61 | 4.30 | 66,821,848 | 11,328.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.538/seresnet152-0430-0f5f3035.npz.log)) |
| SE-PreResNet-10 | 32.35 | 12.20 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.544/sepreresnet10-1220-1bdbe9ae.npz.log)) |
| SE-PreResNet-12 | 31.65 | 11.80 | 5,536,232 | 1,126.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/sepreresnet12-1180-f41eb368.npz.log)) |
| SE-PreResNet-16 | 28.40 | 9.55 | 7,022,976 | 1,589.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/sepreresnet16-0955-904e014f.npz.log)) |
| SE-PreResNet-18 | 27.11 | 8.83 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/sepreresnet18-0883-356877c7.npz.log)) |
| SE-PreResNet-26 | 25.98 | 8.05 | 18,092,188 | 2,747.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/sepreresnet26-0805-e7e6cc56.npz.log)) |
| SE-PreResNet-BC-26b | 22.93 | 6.38 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.399/sepreresnetbc26b-0638-e8393574.npz.log)) |
| SE-PreResNet-BC-38b | 21.46 | 5.66 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.409/sepreresnetbc38b-0566-4b9ce096.npz.log)) |
| SE-PreResNet-50b | 20.75 | 5.31 | 28,080,472 | 4,113.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.461/sepreresnet50b-0531-fde03b26.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 18.73 | 4.35 | 27,559,896 | 4,261.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/seresnext50_32x4d-0435-52a11b61.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.06 | 4.45 | 48,955,416 | 8,012.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.529/seresnext101_32x4d-0445-fd8f34d6.npz.log)) |
| SE-ResNeXt-101 (64x4d) | 18.46 | 4.07 | 88,232,984 | 15,509.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.561/seresnext101_64x4d-0407-5c3b4e4b.npz.log)) |
| SENet-16 | 25.39 | 8.07 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.341/senet16-0807-f45aa3ff.npz.log)) |
| SENet-28 | 21.65 | 5.91 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.356/senet28-0591-7e7bf250.npz.log)) |
| SENet-154 | 18.85 | 4.42 | 115,088,984 | 20,745.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.522/senet154-0442-9f7f0ae3.npz.log)) |
| ResNeSt(A)-BC-14 | 22.28 | 6.33 | 10,611,688 | 2,767.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnestabc14-0633-a76f3b83.npz.log)) |
| ResNeSt(A)-18 | 23.43 | 6.94 | 12,763,784 | 2,587.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnesta18-0694-4ecaf0b7.npz.log)) |
| ResNeSt(A)-BC-26 | 19.59 | 4.69 | 17,069,448 | 3,646.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/resnestabc26-0469-14cb6ca5.npz.log)) |
| ResNeSt(A)-50 | 18.94 | 4.38 | 27,483,240 | 5,403.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.531/resnesta50-0438-84192ed0.npz.log)) |
| ResNeSt(A)-101 | 17.73 | 4.00 | 48,275,016 | 10,247.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta101-0400-bd2efb42.npz.log)) |
| ResNeSt(A)-152 | 18.71 | 4.52 | 65,316,040 | 13,976.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.540/resnesta152-0452-70e32d11.npz.log)) |
| ResNeSt(A)-200 | 16.81 | 3.39 | 70,201,544 | 22,857.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta200-0339-b521427b.npz.log)) |
| ResNeSt(A)-269 | 16.40 | 3.36 | 110,929,480 | 46,012.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta269-0336-933dbe64.npz.log)) |
| AirNet50-1x64d (r=2) | 20.46 | 5.28 | 27,425,864 | 4,772.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.522/airnet50_1x64d_r2-0528-650960af.npz.log)) |
| AirNet50-1x64d (r=16) | 21.11 | 5.47 | 25,714,952 | 4,399.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.519/airnet50_1x64d_r16-0547-6d3d6db4.npz.log)) |
| AirNeXt50-32x4d (r=2) | 19.81 | 5.03 | 27,604,296 | 5,339.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.521/airnext50_32x4d_r2-0503-2bb8e4fc.npz.log)) |
| BAM-ResNet-50 | 20.58 | 5.35 | 25,915,099 | 4,196.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/bam_resnet50-0535-e2069bb5.npz.log)) |
| CBAM-ResNet-50 | 19.95 | 4.88 | 28,089,624 | 4,116.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.537/cbam_resnet50-0488-8a05bbda.npz.log)) |
| SCNet-50 | 21.03 | 5.35 | 25,564,584 | 3,951.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/scnet50-0535-f0ef9a4c.npz.log)) |
| SCNet-101 | 19.68 | 4.64 | 44,565,416 | 7,204.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.507/scnet101-0464-7b157441.npz.log)) |
| SCNet(A)-50 | 19.64 | 4.66 | 25,583,816 | 4,715.84M | From [MCG-NKU/SCNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.472/scneta50-0466-e90cf3c5.npz.log)) |
| RegNetX-200MF | 29.93 | 10.38 | 2,684,792 | 203.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.475/regnetx002-1136-2c208b54.npz.log)) |
| RegNetX-400MF | 26.26 | 8.55 | 5,157,512 | 403.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.479/regnetx004-0855-ecd22778.npz.log)) |
| RegNetX-600MF | 24.67 | 7.60 | 6,196,040 | 608.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx006-0760-fadb78d4.npz.log)) |
| RegNetX-800MF | 24.12 | 7.23 | 7,259,656 | 809.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx008-0723-5fff6491.npz.log)) |
| RegNetX-1.6GF | 22.16 | 6.13 | 9,190,136 | 1,618.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnetx016-0613-5092bfd9.npz.log)) |
| RegNetX-3.2GF | 21.27 | 5.69 | 15,296,552 | 3,199.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/regnetx032-0569-c3625268.npz.log)) |
| RegNetX-4.0GF | 19.53 | 4.71 | 22,118,248 | 3,986.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/regnetx040-0471-362a2ce3.npz.log)) |
| RegNetX-6.4GF | 19.23 | 4.55 | 26,209,256 | 6,491.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.535/regnetx064-0455-3bdbfa0d.npz.log)) |
| RegNetX-8.0GF | 19.62 | 4.67 | 39,572,648 | 8,017.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.515/regnetx080-0467-3618e40b.npz.log)) |
| RegNetX-12GF | 19.96 | 5.19 | 46,106,056 | 12,124.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.542/regnetx120-0519-68699007.npz.log)) |
| RegNetX-16GF | 19.14 | 4.58 | 54,278,536 | 15,986.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.532/regnetx160-0458-99da1e86.npz.log)) |
| RegNetX-32GF | 17.84 | 3.94 | 107,811,560 | 31,790.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.548/regnetx320-0394-22a74cba.npz.log)) |
| RegNetY-200MF | 28.50 | 9.55 | 3,162,996 | 203.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.476/regnety002-0955-5ba3e62c.npz.log)) |
| RegNetY-400MF | 24.84 | 7.52 | 4,344,144 | 410.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/regnety004-0752-e30b7c27.npz.log)) |
| RegNetY-600MF | 23.57 | 7.00 | 6,055,160 | 610.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety006-0700-0917e50c.npz.log)) |
| RegNetY-800MF | 22.56 | 6.45 | 6,263,168 | 808.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety008-0645-aa4c6104.npz.log)) |
| RegNetY-1.6GF | 21.24 | 5.71 | 11,202,430 | 1,629.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnety016-0571-962bc21c.npz.log)) |
| RegNetY-3.2GF | 18.31 | 4.11 | 19,436,338 | 3,199.15M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety032-0411-7097f659.npz.log)) |
| RegNetY-4.0GF | 19.52 | 4.66 | 20,646,656 | 3,999.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.494/regnety040-0466-c84f52df.npz.log)) |
| RegNetY-6.4GF | 18.91 | 4.46 | 30,583,252 | 6,388.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.513/regnety064-0446-22d3ab05.npz.log)) |
| RegNetY-8.0GF | 18.78 | 4.37 | 39,180,068 | 7,996.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.516/regnety080-0437-f324b046.npz.log)) |
| RegNetY-12GF | 18.49 | 4.30 | 51,822,544 | 12,132.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.526/regnety120-0430-fa03425a.npz.log)) |
| RegNetY-16GF | 18.69 | 4.29 | 83,590,140 | 15,944.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.527/regnety160-0429-0a034eb9.npz.log)) |
| RegNetY-32GF | 17.77 | 3.72 | 145,046,770 | 32,317.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.550/regnety320-0372-7bc53739.npz.log)) |
| PyramidNet-101 (a=360) | 20.43 | 5.20 | 42,455,070 | 8,743.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.507/pyramidnet101_a360-0520-e37960c4.npz.log)) |
| DiracNetV2-18 | 30.60 | 11.13 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1113-b85b43d1.npz.log)) |
| DiracNetV2-34 | 27.90 | 9.48 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0948-0245163a.npz.log)) |
| DenseNet-121 | 23.24 | 6.83 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.314/densenet121-0683-4caa2458.npz.log)) |
| DenseNet-161 | 21.79 | 5.90 | 28,681,000 | 7,793.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.432/densenet161-0590-a514f930.npz.log)) |
| DenseNet-169 | 22.11 | 6.09 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.406/densenet169-0609-99c9bddf.npz.log)) |
| DenseNet-201 | 21.56 | 5.90 | 20,013,928 | 4,347.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.426/densenet201-0590-f50cfbb1.npz.log)) |
| CondenseNet-74 (C=G=4) | 26.81 | 8.61 | 4,773,944 | 546.06M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c4_g4-0861-ef6077ec.npz.log)) |
| CondenseNet-74 (C=G=8) | 29.74 | 10.43 | 2,935,416 | 291.52M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c8_g8-1043-277fbfb8.npz.log)) |
| PeleeNet | 29.38 | 9.84 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.496/peleenet-0984-32d512c0.npz.log)) |
| WRN-50-2 | 22.07 | 6.07 | 68,849,128 | 11,405.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.520/wrn50_2-0607-c5976365.npz.log)) |
| DRN-C-26 | 24.34 | 7.10 | 21,126,584 | 16,993.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.508/drnc26-0710-16a6251b.npz.log)) |
| DRN-C-42 | 22.31 | 6.14 | 31,234,744 | 25,093.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.556/drnc42-0614-6b2099e0.npz.log)) |
| DRN-C-58 | 20.47 | 5.17 | 40,542,008 | 32,489.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.559/drnc58-0517-a333394a.npz.log)) |
| DRN-D-22 | 24.73 | 7.47 | 16,393,752 | 13,051.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.498/drnd22-0747-e4c5cf73.npz.log)) |
| DRN-D-38 | 22.83 | 6.24 | 26,501,912 | 21,151.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.552/drnd38-0624-1a1a7b99.npz.log)) |
| DRN-D-54 | 20.31 | 5.00 | 35,809,176 | 28,547.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.554/drnd54-0500-b679707c.npz.log)) |
| DRN-D-105 | 19.76 | 4.89 | 54,801,304 | 43,442.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.564/drnd105-0489-49cba09d.npz.log)) |
| DPN-68 | 22.93 | 6.56 | 12,611,602 | 2,351.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.310/dpn68-0656-bf9b72e9.npz.log)) |
| DPN-98 | 18.31 | 4.26 | 61,570,728 | 11,716.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.540/dpn98-0426-2ab67669.npz.log)) |
| DPN-131 | 19.38 | 4.79 | 79,254,504 | 16,076.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.534/dpn131-0479-eb5a4bc2.npz.log)) |
| DarkNet Tiny | 40.33 | 17.46 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-b04fa463.npz.log)) |
| DarkNet Ref | 38.09 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1671-b2d5721f.npz.log)) |
| DarkNet-53 | 21.27 | 5.52 | 41,609,928 | 7,133.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.501/darknet53-0552-3e6f4076.npz.log)) |
| i-RevNet-301 | 24.27 | 7.38 | 125,120,356 | 14,453.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.564/irevnet301-0738-cc2ac933.npz.log)) |
| BagNet-9 | 48.78 | 25.44 | 15,688,744 | 16,049.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.553/bagnet9-2544-346ee143.npz.log)) |
| BagNet-17 | 36.51 | 15.23 | 16,213,032 | 15,768.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.558/bagnet17-1523-ccc69ea4.npz.log)) |
| BagNet-33 | 29.45 | 10.44 | 18,310,184 | 16,371.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.561/bagnet33-1044-17d82bc6.npz.log)) |
| DLA-34 | 24.39 | 7.06 | 15,742,104 | 3,071.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/dla34-0706-576dd492.npz.log)) |
| DLA-46-C | 33.85 | 12.92 | 1,301,400 | 585.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.282/dla46c-1292-98e3efd5.npz.log)) |
| DLA-X-46-C | 32.90 | 12.28 | 1,068,440 | 546.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.293/dla46xc-1228-c2dc61bc.npz.log)) |
| DLA-60 | 21.26 | 5.54 | 22,036,632 | 4,255.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.494/dla60-0554-740881ca.npz.log)) |
| DLA-X-60 | 20.70 | 5.54 | 17,352,344 | 3,543.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/dla60x-0554-4d757562.npz.log)) |
| DLA-X-60-C | 30.70 | 10.76 | 1,319,832 | 596.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.289/dla60xc-1076-4c418399.npz.log)) |
| DLA-102 | 20.54 | 5.18 | 33,268,888 | 7,190.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/dla102-0518-ebbfedd7.npz.log)) |
| DLA-X-102 | 19.56 | 4.74 | 26,309,272 | 5,884.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.528/dla102x-0474-39bad3b5.npz.log)) |
| DLA-X2-102 | 18.68 | 4.25 | 41,282,200 | 9,340.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.542/dla102x2-0425-4ffec225.npz.log)) |
| DLA-169 | 19.27 | 4.61 | 53,389,720 | 11,593.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.539/dla169-0461-f47952e2.npz.log)) |
| FishNet-150 | 19.14 | 4.65 | 24,959,400 | 6,435.05M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.502/fishnet150-0465-724e74fc.npz.log)) |
| ESPNetv2 x0.5 | 43.42 | 20.79 | 1,241,332 | 35.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.562/espnetv2_wd2-2079-f77e3ecf.npz.log)) |
| ESPNetv2 x1.0 | 35.30 | 14.31 | 1,670,072 | 98.09M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_w1-1431-eab8d605.npz.log)) |
| ESPNetv2 x1.25 | 32.24 | 12.24 | 1,965,440 | 138.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.564/espnetv2_w5d4-1224-68e58dc6.npz.log)) |
| ESPNetv2 x1.5 | 31.04 | 11.40 | 2,314,856 | 185.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.520/espnetv2_w3d2-1140-1bfaa23b.npz.log)) |
| ESPNetv2 x2.0 | 29.04 | 9.64 | 3,498,136 | 306.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.565/espnetv2_w2-0964-07819e61.npz.log)) |
| DiCENet x0.2 | 54.93 | 30.55 | 1,130,704 | 18.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.563/dicenet_wd5-3055-708911b8.npz.log)) |
| DiCENet x0.5 | 47.19 | 23.09 | 1,214,120 | 30.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.561/dicenet_wd2-2309-b383f3a6.npz.log)) |
| DiCENet x0.75 | 38.19 | 16.42 | 1,495,676 | 55.80M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w3d4-1642-c97e9157.npz.log)) |
| DiCENet x1.0 | 35.02 | 14.13 | 1,805,604 | 82.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.513/dicenet_w1-1413-e926dbb2.npz.log)) |
| DiCENet x1.25 | 33.08 | 12.55 | 2,162,888 | 111.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.515/dicenet_w5d4-1255-f4914cf7.npz.log)) |
| DiCENet x1.5 | 30.95 | 11.44 | 2,652,200 | 151.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.522/dicenet_w3d2-1144-a6b89c5e.npz.log)) |
| DiCENet x1.75 | 30.14 | 10.80 | 3,264,932 | 201.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.527/dicenet_w7d8-1080-f2a3e6ae.npz.log)) |
| DiCENet x2.0 | 29.92 | 10.63 | 3,979,044 | 257.95M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w2-1063-224f4153.npz.log)) |
| HRNet-W18 Small V1 | 26.24 | 8.73 | 13,187,464 | 1,615.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/hrnet_w18_small_v1-0873-96476e4b.npz.log)) |
| HRNet-W18 Small V2 | 21.73 | 6.03 | 15,597,464 | 2,618.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/hrnet_w18_small_v2-0603-87be5731.npz.log)) |
| HRNetV2-W18 | 20.12 | 5.04 | 21,299,004 | 4,323.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.508/hrnetv2_w18-0504-27b854bf.npz.log)) |
| HRNetV2-W30 | 20.26 | 5.06 | 37,712,220 | 8,156.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.525/hrnetv2_w30-0506-04453d82.npz.log)) |
| HRNetV2-W32 | 19.93 | 4.93 | 41,232,680 | 8,974.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.528/hrnetv2_w32-0493-9f43468d.npz.log)) |
| HRNetV2-W40 | 19.66 | 4.79 | 57,557,160 | 12,752.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.534/hrnetv2_w40-0479-738507ee.npz.log)) |
| HRNetV2-W44 | 19.67 | 4.90 | 67,064,984 | 14,946.96M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.541/hrnetv2_w44-0490-6e10e2b5.npz.log)) |
| HRNetV2-W48 | 19.45 | 4.86 | 77,469,864 | 17,345.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.541/hrnetv2_w48-0486-e8df81b5.npz.log)) |
| HRNetV2-W64 | 19.50 | 4.79 | 128,059,944 | 28,976.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/hrnetv2_w64-0479-702c5688.npz.log)) |
| VoVNet-27-slim | 29.35 | 9.78 | 3,525,736 | 2,187.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.551/vovnet27s-0978-6983e4ee.npz.log)) |
| VoVNet-39 | 21.49 | 5.53 | 22,600,296 | 7,086.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/vovnet39-0553-6a8b6783.npz.log)) |
| VoVNet-57 | 20.16 | 5.12 | 36,640,296 | 8,943.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/vovnet57-0512-beb31bd5.npz.log)) |
| SelecSLS-42b | 21.74 | 6.01 | 32,458,248 | 2,980.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/selecsls42b-0601-d89a5042.npz.log)) |
| SelecSLS-60 | 20.18 | 5.14 | 30,670,768 | 3,591.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.496/selecsls60-0514-637c2506.npz.log)) |
| SelecSLS-60b | 20.60 | 5.38 | 32,774,064 | 3,629.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/selecsls60b-0538-f9f8d657.npz.log)) |
| HarDNet-39DS | 26.48 | 8.70 | 3,488,228 | 437.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/hardnet39ds-0870-fcf92ed6.npz.log)) |
| HarDNet-68DS | 24.22 | 7.43 | 4,180,602 | 788.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.487/hardnet68ds-0743-a6b77ed0.npz.log)) |
| HarDNet-68 | 23.97 | 7.04 | 17,565,348 | 4,256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.557/hardnet68-0704-71176051.npz.log)) |
| HarDNet-85 | 21.83 | 5.69 | 36,670,212 | 9,088.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/hardnet85-0569-5f3cb7be.npz.log)) |
| SqueezeNet v1.0 | 38.76 | 17.38 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1738-4c55a6a5.npz.log)) |
| SqueezeNet v1.1 | 39.13 | 17.40 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1740-b236c204.npz.log)) |
| SqueezeResNet v1.0 | 39.36 | 17.66 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.178/squeezeresnet_v1_0-1766-6dc69dc2.npz.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.87 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1787-f40e6051.npz.log)) |
| 1.0-SqNxt-23 | 42.62 | 19.03 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-1903-ef3d725b.npz.log)) |
| 1.0-SqNxt-23v5 | 40.96 | 17.86 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-1786-8b24c6e3.npz.log)) |
| 1.5-SqNxt-23 | 34.71 | 13.44 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.210/sqnxt23_w3d2-1344-a5c3b21e.npz.log)) |
| 1.5-SqNxt-23v5 | 33.79 | 12.92 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.212/sqnxt23v5_w3d2-1292-c997e279.npz.log)) |
| 2.0-SqNxt-23 | 30.43 | 10.82 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.240/sqnxt23_w2-1082-cf7aebef.npz.log)) |
| 2.0-SqNxt-23v5 | 29.58 | 10.43 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.216/sqnxt23v5_w2-1043-e9e849cd.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.04 | 36.81 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3681-15d3e787.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.30 | 36.16 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3616-064f7f7f.npz.log)) |
| ShuffleNet x0.5 (g=1) | 46.24 | 22.35 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.174/shufflenet_g1_wd2-2235-5d83cc28.npz.log)) |
| ShuffleNet x0.5 (g=3) | 43.83 | 20.61 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2061-557e4397.npz.log)) |
| ShuffleNet x0.75 (g=1) | 39.26 | 16.77 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.218/shufflenet_g1_w3d4-1677-b5515ea9.npz.log)) |
| ShuffleNet x0.75 (g=3) | 37.83 | 16.13 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.219/shufflenet_g3_w3d4-1613-55129cb5.npz.log)) |
| ShuffleNet x1.0 (g=1) | 34.44 | 13.48 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.223/shufflenet_g1_w1-1348-37cc6c5f.npz.log)) |
| ShuffleNet x1.0 (g=2) | 33.94 | 13.33 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.241/shufflenet_g2_w1-1333-e473c62f.npz.log)) |
| ShuffleNet x1.0 (g=3) | 33.99 | 13.26 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.244/shufflenet_g3_w1-1326-95df0487.npz.log)) |
| ShuffleNet x1.0 (g=4) | 33.87 | 13.08 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.245/shufflenet_g4_w1-1308-8ed92f35.npz.log)) |
| ShuffleNet x1.0 (g=8) | 33.68 | 13.21 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.250/shufflenet_g8_w1-1321-2fea8945.npz.log)) |
| ShuffleNetV2 x0.5 | 43.45 | 20.73 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-2073-c5e5a23c.npz.log)) |
| ShuffleNetV2 x1.0 | 33.39 | 12.98 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1298-3830a2da.npz.log)) |
| ShuffleNetV2 x1.5 | 28.87 | 10.14 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.288/shufflenetv2_w3d2-1014-5f75edb1.npz.log)) |
| ShuffleNetV2 x2.0 | 27.01 | 8.99 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.301/shufflenetv2_w2-0899-a44b1d5d.npz.log)) |
| ShuffleNetV2b x0.5 | 39.78 | 17.87 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.157/shufflenetv2b_wd2-1787-08a12021.npz.log)) |
| ShuffleNetV2b x1.0 | 30.36 | 11.00 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.161/shufflenetv2b_w1-1100-21562fb2.npz.log)) |
| ShuffleNetV2b x1.5 | 26.92 | 8.78 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.203/shufflenetv2b_w3d2-0878-7a5c7ed4.npz.log)) |
| ShuffleNetV2b x2.0 | 25.23 | 8.10 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.242/shufflenetv2b_w2-0810-636e281c.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.42 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2042-9e3ff283.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.07 | 19.19 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1919-f6fd56fa.npz.log)) |
| 160-MENet-8x1 (g=8) | 43.54 | 20.42 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2042-250fd765.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.86 | 13.01 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1301-39c25ca3.npz.log)) |
| 256-MENet-12x1 (g=4) | 32.30 | 12.18 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1218-57160b09.npz.log)) |
| 348-MENet-12x1 (g=3) | 27.86 | 9.36 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.173/menet348_12x1_g3-0936-ee7e056d.npz.log)) |
| 352-MENet-12x1 (g=8) | 31.28 | 11.72 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.198/menet352_12x1_g8-1172-c256ae25.npz.log)) |
| 456-MENet-24x1 (g=3) | 25.07 | 7.79 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.237/menet456_24x1_g3-0779-5af355f6.npz.log)) |
| MobileNet x0.25 | 45.85 | 22.16 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2216-09c50ab8.npz.log)) |
| MobileNet x0.5 | 33.89 | 13.37 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1337-48d12ee3.npz.log)) |
| MobileNet x0.75 | 29.86 | 10.53 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1053-d7ec3192.npz.log)) |
| MobileNet x1.0 | 26.47 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0866-b888f817.npz.log)) |
| MobileNetb x0.25 | 45.22 | 21.64 | 467,592 | 42.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_wd4-2164-65e4eeb5.npz.log)) |
| MobileNetb x0.5 | 32.93 | 12.69 | 1,326,632 | 153.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.480/mobilenetb_wd2-1269-a649a585.npz.log)) |
| MobileNetb x0.75 | 29.11 | 10.19 | 2,578,120 | 330.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_w3d4-1019-a54016b2.npz.log)) |
| MobileNetb x1.0 | 25.07 | 7.88 | 4,222,056 | 574.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/mobilenetb_w1-0788-e95ffdb9.npz.log)) |
| FD-MobileNet x0.25 | 55.43 | 30.63 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.177/fdmobilenet_wd4-3063-55407f3a.npz.log)) |
| FD-MobileNet x0.5 | 42.68 | 19.76 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1976-6299d442.npz.log)) |
| FD-MobileNet x0.75 | 37.94 | 15.99 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1599-cdfc2e04.npz.log)) |
| FD-MobileNet x1.0 | 33.90 | 13.16 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1316-0ed6f00c.npz.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.11 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2411-9fc398d3.npz.log)) |
| MobileNetV2 x0.5 | 35.56 | 14.44 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1444-ca0906e1.npz.log)) |
| MobileNetV2 x0.75 | 29.75 | 10.47 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.230/mobilenetv2_w3d4-1047-a25fd26c.npz.log)) |
| MobileNetV2 x1.0 | 26.80 | 8.66 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.213/mobilenetv2_w1-0866-efc3331e.npz.log)) |
| MobileNetV2b x0.25 | 46.71 | 23.42 | 1,516,312 | 33.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_wd4-2342-bf23c314.npz.log)) |
| MobileNetV2b x0.5 | 34.25 | 13.76 | 1,964,448 | 96.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mobilenetv2b_wd2-1376-f68cc37d.npz.log)) |
| MobileNetV2b x0.75 | 30.19 | 10.67 | 2,626,968 | 190.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w3d4-1067-ba0caa95.npz.log)) |
| MobileNetV2b x1.0 | 27.19 | 8.90 | 3,503,872 | 315.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w1-0890-dbc98d15.npz.log)) |
| MobileNetV3 L/224/1.0 | 24.36 | 7.33 | 5,481,752 | 227.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/mobilenetv3_large_w1-0733-20f2980c.npz.log)) |
| IGCV3 x0.25 | 53.36 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2828-25942192.npz.log)) |
| IGCV3 x0.5 | 39.36 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-86246558.npz.log)) |
| IGCV3 x0.75 | 30.67 | 10.99 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.207/igcv3_w3d4-1099-b0dbc54a.npz.log)) |
| IGCV3 x1.0 | 27.70 | 8.98 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.243/igcv3_w1-0898-5fd85acd.npz.log)) |
| MnasNet-B1 | 24.67 | 7.25 | 4,383,312 | 326.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mnasnet_b1-0725-2733981b.npz.log)) |
| MnasNet-A1 | 24.09 | 7.05 | 3,887,038 | 326.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mnasnet_a1-0705-9ac62ab0.npz.log)) |
| DARTS | 24.96 | 7.58 | 4,718,752 | 539.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/darts-0758-8085336b.npz.log)) |
| ProxylessNAS CPU | 24.76 | 7.52 | 4,361,648 | 459.96M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.324/proxylessnas_cpu-0752-22bd211b.npz.log)) |
| ProxylessNAS GPU | 24.62 | 7.23 | 7,119,848 | 476.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.333/proxylessnas_gpu-0723-b81256a1.npz.log)) |
| ProxylessNAS Mobile | 25.31 | 7.85 | 4,080,512 | 332.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.326/proxylessnas_mobile-0785-561f3416.npz.log)) |
| ProxylessNAS Mob-14 | 22.95 | 6.51 | 6,857,568 | 597.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.331/proxylessnas_mobile14-0651-7467ce2d.npz.log)) |
| FBNet-Cb | 24.77 | 7.64 | 5,572,200 | 399.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/fbnet_cb-0764-9a8153a5.npz.log)) |
| Xception | 20.36 | 5.18 | 22,855,952 | 8,403.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.544/xception-0518-a311fd37.npz.log)) |
| InceptionV3 | 20.47 | 5.35 | 23,834,568 | 5,743.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.552/inceptionv3-0535-1662fcdc.npz.log)) |
| InceptionV4 | 19.94 | 4.87 | 42,679,816 | 12,304.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.543/inceptionv4-0487-75970908.npz.log)) |
| InceptionResNetV1 | 19.58 | 4.81 | 23,995,624 | 6,329.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.552/inceptionresnetv1-0481-a3ddee2c.npz.log)) |
| InceptionResNetV2 | 19.44 | 4.72 | 55,843,464 | 13,188.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.547/inceptionresnetv2-0472-178ff37a.npz.log)) |
| PolyNet | 19.08 | 4.50 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0450-6dc7028b.npz.log)) |
| NASNet-A 4@1056 | 25.18 | 7.90 | 5,289,978 | 584.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/nasnet_4a1056-0790-92b4789b.npz.log)) |
| NASNet-A 6@4032 | 18.17 | 4.22 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0422-d49d4663.npz.log)) |
| PNASNet-5-Large | 17.90 | 4.26 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0426-3c2755dc.npz.log)) |
| SPNASNet | 25.09 | 7.79 | 4,421,616 | 346.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/spnasnet-0779-4fa174db.npz.log)) |
| EfficientNet-B0 | 24.48 | 7.25 | 5,288,548 | 414.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.364/efficientnet_b0-0725-8d6f1744.npz.log)) |
| EfficientNet-B1 | 23.02 | 6.33 | 7,794,184 | 732.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.376/efficientnet_b1-0633-4ac377d9.npz.log)) |
| EfficientNet-B0b | 22.97 | 6.69 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b0b-0669-366e9c54.npz.log)) |
| EfficientNet-B1b | 20.97 | 5.67 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b1b-0567-2826a686.npz.log)) |
| EfficientNet-B2b | 20.02 | 5.14 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b2b-0514-93c91747.npz.log)) |
| EfficientNet-B3b | 18.72 | 4.36 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b3b-0436-82eb9d91.npz.log)) |
| EfficientNet-B4b | 17.46 | 3.92 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b4b-0392-81138451.npz.log)) |
| EfficientNet-B5b | 16.52 | 3.39 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b5b-0339-fb684f5d.npz.log)) |
| EfficientNet-B6b | 16.31 | 3.24 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b6b-0324-acaad4db.npz.log)) |
| EfficientNet-B7b | 15.92 | 3.23 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b7b-0323-031b7bd5.npz.log)) |
| EfficientNet-B0c* | 22.54 | 6.44 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b0c-0644-e95e873d.npz.log)) |
| EfficientNet-B1c* | 20.56 | 5.57 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b1c-0557-07796241.npz.log)) |
| EfficientNet-B2c* | 19.65 | 4.96 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b2c-0496-5a0d3333.npz.log)) |
| EfficientNet-B3c* | 18.24 | 4.40 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b3c-0440-ec082c31.npz.log)) |
| EfficientNet-B4c* | 16.86 | 3.68 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b4c-0368-c025d233.npz.log)) |
| EfficientNet-B5c* | 15.91 | 3.11 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b5c-0311-e01810a9.npz.log)) |
| EfficientNet-B6c* | 15.49 | 2.98 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b6c-0298-72ac53f6.npz.log)) |
| EfficientNet-B7c* | 15.18 | 2.91 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b7c-0291-c0711f21.npz.log)) |
| EfficientNet-B8c* | 14.83 | 2.76 | 87,413,142 | 64,541.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b8c-0276-d1c7aa15.npz.log)) |
| EfficientNet-Edge-Small-b* | 22.48 | 6.29 | 5,438,392 | 2,378.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_small_b-0629-4aac3591.npz.log)) |
| EfficientNet-Edge-Medium-b* | 21.06 | 5.52 | 6,899,496 | 3,700.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_medium_b-0552-fdf98bd5.npz.log)) |
| EfficientNet-Edge-Large-b* | 19.58 | 4.89 | 10,589,712 | 9,747.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_large_b-0489-45f05958.npz.log)) |
| MixNet-S | 23.77 | 7.05 | 4,134,606 | 260.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_s-0705-4822e76d.npz.log)) |
| MixNet-M | 22.39 | 6.34 | 5,014,382 | 366.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_m-0634-2638a388.npz.log)) |
| MixNet-L | 21.43 | 5.59 | 7,329,252 | 591.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.500/mixnet_l-0559-ff6929ef.npz.log)) |
| ResNet(A)-10 | 30.94 | 11.61 | 5,438,024 | 1,135.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.484/resneta10-1161-c28d6ca6.npz.log)) |
| ResNet(A)-BC-14 | 27.72 | 9.57 | 10,084,168 | 1,721.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.477/resnetabc14b-0957-84e05fea.npz.log)) |
| ResNet(A)-18 | 25.40 | 8.05 | 11,708,744 | 2,062.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/resneta18-0805-f4088383.npz.log)) |
| ResNet(A)-50b | 20.85 | 5.37 | 25,576,264 | 4,352.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/resneta50b-0537-204eb60e.npz.log)) |
| ResNet(A)-101b | 19.03 | 4.45 | 44,568,392 | 8,072.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.532/resneta101b-0445-5b214e8c.npz.log)) |
| ResNet(A)-152b | 18.55 | 4.23 | 60,212,040 | 11,796.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.524/resneta152b-0423-515d6b62.npz.log)) |
| ResNet(D)-50b | 20.80 | 5.50 | 25,680,808 | 20,497.60M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd50b-0550-7ba88f04.npz.log)) |
| ResNet(D)-101b | 19.49 | 4.60 | 44,672,936 | 35,392.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd101b-0460-b90f971e.npz.log)) |
| ResNet(D)-152b | 19.36 | 4.70 | 60,316,584 | 47,662.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd152b-0470-41442334.npz.log)) |

### CIFAR-10

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 7.43 | 966,986 | 222.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.175/nin_cifar10-0743-045abfde.npz.log)) |
| ResNet-20 | 5.97 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet20_cifar10-0597-15145d2e.npz.log)) |
| ResNet-56 | 4.52 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet56_cifar10-0452-eb7923aa.npz.log)) |
| ResNet-110 | 3.69 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet110_cifar10-0369-27d76fce.npz.log)) |
| ResNet-164(BN) | 3.68 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.179/resnet164bn_cifar10-0368-d8659366.npz.log)) |
| ResNet-272(BN) | 3.33 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_cifar10-0333-b7c6902a.npz.log)) |
| ResNet-542(BN) | 3.43 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_cifar10-0343-b6598e7a.npz.log)) |
| ResNet-1001 | 3.28 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.201/resnet1001_cifar10-0328-0e27556c.npz.log)) |
| ResNet-1202 | 3.53 | 19,424,026 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.214/resnet1202_cifar10-0353-d82bb435.npz.log)) |
| PreResNet-20 | 6.51 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet20_cifar10-0651-5cf94722.npz.log)) |
| PreResNet-56 | 4.49 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet56_cifar10-0449-73ea193a.npz.log)) |
| PreResNet-110 | 3.86 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet110_cifar10-0386-544ed0f0.npz.log)) |
| PreResNet-164(BN) | 3.64 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.196/preresnet164bn_cifar10-0364-c0ff2438.npz.log)) |
| PreResNet-272(BN) | 3.25 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_cifar10-0325-8f8f375d.npz.log)) |
| PreResNet-542(BN) | 3.14 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_cifar10-0314-86a2b5f5.npz.log)) |
| PreResNet-1001 | 2.65 | 10,327,706 | 1,536.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.209/preresnet1001_cifar10-0265-1f3028bd.npz.log)) |
| PreResNet-1202 | 3.39 | 19,423,834 | 2,857.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.246/preresnet1202_cifar10-0339-cc2bd85a.npz.log)) |
| ResNeXt-29 (32x4d) | 3.15 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.169/resnext29_32x4d_cifar10-0315-442eca6c.npz.log)) |
| ResNeXt-29 (16x64d) | 2.41 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.176/resnext29_16x64d_cifar10-0241-e80d3cb5.npz.log)) |
| ResNeXt-272 (1x64d) | 2.55 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_cifar10-0255-1ca66300.npz.log)) |
| ResNeXt-272 (2x32d) | 2.74 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_cifar10-0274-94e492a4.npz.log)) |
| SE-ResNet-20 | 6.01 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_cifar10-0601-143eba2a.npz.log)) |
| SE-ResNet-56 | 4.13 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_cifar10-0413-66486cdb.npz.log)) |
| SE-ResNet-110 | 3.63 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_cifar10-0363-9a85ff95.npz.log)) |
| SE-ResNet-164(BN) | 3.39 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_cifar10-0339-4c59e76f.npz.log)) |
| SE-ResNet-272(BN) | 3.39 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_cifar10-0339-8081d1be.npz.log)) |
| SE-ResNet-542(BN) | 3.47 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_cifar10-0347-e67d0c05.npz.log)) |
| SE-PreResNet-20 | 6.18 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_cifar10-0618-cbc1c4df.npz.log)) |
| SE-PreResNet-56 | 4.51 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_cifar10-0451-0b34942c.npz.log)) |
| SE-PreResNet-110 | 4.54 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_cifar10-0454-4c062f46.npz.log)) |
| SE-PreResNet-164(BN) | 3.73 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_cifar10-0373-e82ad7ff.npz.log)) |
| SE-PreResNet-272(BN) | 3.39 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_cifar10-0339-02e14113.npz.log)) |
| SE-PreResNet-542(BN) | 3.08 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_cifar10-0308-1e726874.npz.log)) |
| PyramidNet-110 (a=48) | 3.72 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.184/pyramidnet110_a48_cifar10-0372-965fce37.npz.log)) |
| PyramidNet-110 (a=84) | 2.98 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.185/pyramidnet110_a84_cifar10-0298-7b38a0f6.npz.log)) |
| PyramidNet-110 (a=270) | 2.51 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.194/pyramidnet110_a270_cifar10-0251-b3456ddd.npz.log)) |
| PyramidNet-164 (a=270, BN) | 2.42 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.264/pyramidnet164_a270_bn_cifar10-0242-783e21b5.npz.log)) |
| PyramidNet-200 (a=240, BN) | 2.44 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.268/pyramidnet200_a240_bn_cifar10-0244-89ae1856.npz.log)) |
| PyramidNet-236 (a=220, BN) | 2.47 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.285/pyramidnet236_a220_bn_cifar10-0247-6b9a2966.npz.log)) |
| PyramidNet-272 (a=200, BN) | 2.39 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.284/pyramidnet272_a200_bn_cifar10-0239-533f8d89.npz.log)) |
| DenseNet-40 (k=12) | 5.61 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.193/densenet40_k12_cifar10-0561-a37df881.npz.log)) |
| DenseNet-BC-40 (k=12) | 6.43 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.231/densenet40_k12_bc_cifar10-0643-234918e7.npz.log)) |
| DenseNet-BC-40 (k=24) | 4.52 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.220/densenet40_k24_bc_cifar10-0452-3ec459af.npz.log)) |
| DenseNet-BC-40 (k=36) | 4.04 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.224/densenet40_k36_bc_cifar10-0404-6be4225a.npz.log)) |
| DenseNet-100 (k=12) | 3.66 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.205/densenet100_k12_cifar10-0366-85031735.npz.log)) |
| DenseNet-100 (k=24) | 3.13 | 16,114,138 | 5,354.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.252/densenet100_k24_cifar10-0313-939ef309.npz.log)) |
| DenseNet-BC-100 (k=12) | 4.16 | 769,162 | 298.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.189/densenet100_k12_bc_cifar10-0416-160a0641.npz.log)) |
| DenseNet-BC-190 (k=40) | 2.52 | 25,624,430 | 9,400.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.286/densenet190_k40_bc_cifar10-0252-57f2fa70.npz.log)) |
| DenseNet-BC-250 (k=24) | 2.67 | 15,324,406 | 5,519.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.290/densenet250_k24_bc_cifar10-0267-03b26887.npz.log)) |
| X-DenseNet-BC-40-2 (k=24) | 5.31 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.226/xdensenet40_2_k24_bc_cifar10-0531-d3c448ab.npz.log)) |
| X-DenseNet-BC-40-2 (k=36) | 4.37 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.233/xdensenet40_2_k36_bc_cifar10-0437-fb6d7431.npz.log)) |
| WRN-16-10 | 2.93 | 17,116,634 | 2,414.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn16_10_cifar10-0293-4ac60015.npz.log)) |
| WRN-28-10 | 2.39 | 36,479,194 | 5,246.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn28_10_cifar10-0239-f8a24941.npz.log)) |
| WRN-40-8 | 2.37 | 35,748,314 | 5,176.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn40_8_cifar10-0237-3f56f24a.npz.log)) |
| WRN-20-10-1bit | 3.26 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_cifar10-0326-3288c59a.npz.log)) |
| WRN-20-10-32bit | 3.14 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_cifar10-0314-90b3fc15.npz.log)) |
| RoR-3-56 | 5.43 | 762,746 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.228/ror3_56_cifar10-0543-7ca1b24c.npz.log)) |
| RoR-3-110 | 4.35 | 1,637,690 | 242.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.235/ror3_110_cifar10-0435-bf021f25.npz.log)) |
| RoR-3-164 | 3.93 | 2,512,634 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_cifar10-0393-7ac7b446.npz.log)) |
| RiR | 3.28 | 9,492,980 | 1,281.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_cifar10-0328-9780c77d.npz.log)) |
| Shake-Shake-ResNet-20-2x16d | 5.15 | 541,082 | 81.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.215/shakeshakeresnet20_2x16d_cifar10-0515-e2f524b5.npz.log)) |
| Shake-Shake-ResNet-26-2x32d | 3.17 | 2,923,162 | 428.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.217/shakeshakeresnet26_2x32d_cifar10-0317-5422fce1.npz.log)) |
| DIA-ResNet-20 | 6.22 | 286,866 | 41.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet20_cifar10-0622-1c5f4c8a.npz.log)) |
| DIA-ResNet-56 | 5.05 | 870,162 | 129.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet56_cifar10-0505-4073bb0c.npz.log)) |
| DIA-ResNet-110 | 4.10 | 1,745,106 | 264.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet110_cifar10-0410-5d051745.npz.log)) |
| DIA-ResNet-164(BN) | 3.50 | 1,923,002 | 343.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet164bn_cifar10-0350-27cfe80d.npz.log)) |
| DIA-PreResNet-20 | 6.42 | 286,674 | 41.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_cifar10-0642-bfcfd5c6.npz.log)) |
| DIA-PreResNet-56 | 4.83 | 869,970 | 129.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_cifar10-0483-d5229916.npz.log)) |
| DIA-PreResNet-110 | 4.25 | 1,744,914 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_cifar10-0425-9fab76b9.npz.log)) |
| DIA-PreResNet-164(BN) | 3.56 | 1,922,106 | 343.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_cifar10-0356-7a0b1243.npz.pth.log)) |

### CIFAR-100

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 28.39 | 984,356 | 224.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.183/nin_cifar100-2839-89104763.npz.log)) |
| ResNet-20 | 29.64 | 278,324 | 41.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.180/resnet20_cifar100-2964-6a85f07e.npz.log)) |
| ResNet-56 | 24.88 | 861,620 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.181/resnet56_cifar100-2488-2d641cde.npz.log)) |
| ResNet-110 | 22.80 | 1,736,564 | 255.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.190/resnet110_cifar100-2280-d2ec4ff1.npz.log)) |
| ResNet-164(BN) | 20.44 | 1,727,284 | 255.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.182/resnet164bn_cifar100-2044-190ab6b4.npz.log)) |
| ResNet-272(BN) | 20.07 | 2,840,116 | 420.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_cifar100-2007-fe6b27f8.npz.log)) |
| ResNet-542(BN) | 19.32 | 5,622,196 | 833.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_cifar100-1932-4f95b380.npz.log)) |
| ResNet-1001 | 19.79 | 10,351,732 | 1,536.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.254/resnet1001_cifar100-1979-6416c8d2.npz.log)) |
| ResNet-1202 | 21.56 | 19,429,876 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.410/resnet1202_cifar100-2156-71113602.npz.log)) |
| PreResNet-20 | 30.22 | 278,132 | 41.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.187/preresnet20_cifar100-3022-e3fd9391.npz.log)) |
| PreResNet-56 | 25.05 | 861,428 | 127.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.188/preresnet56_cifar100-2505-f879fb4e.npz.log)) |
| PreResNet-110 | 22.67 | 1,736,372 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.191/preresnet110_cifar100-2267-4e010af0.npz.log)) |
| PreResNet-164(BN) | 20.18 | 1,726,388 | 255.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.192/preresnet164bn_cifar100-2018-5228dfbd.npz.log)) |
| PreResNet-272(BN) | 19.63 | 2,839,220 | 420.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_cifar100-1963-52a0ebab.npz.log)) |
| PreResNet-542(BN) | 18.71 | 5,621,300 | 833.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_cifar100-1871-d7343a66.npz.log)) |
| PreResNet-1001 | 18.41 | 10,350,836 | 1,536.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.283/preresnet1001_cifar100-1841-fcbddbdb.npz.log)) |
| ResNeXt-29 (32x4d) | 19.50 | 4,868,004 | 780.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.200/resnext29_32x4d_cifar100-1950-de139852.npz.log)) |
| ResNeXt-29 (16x64d) | 16.93 | 68,247,460 | 10,709.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.322/resnext29_16x64d_cifar100-1693-762f79b3.npz.log)) |
| ResNeXt-272 (1x64d) | 19.11 | 44,632,996 | 6,565.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_cifar100-1911-9a9b397c.npz.log)) |
| ResNeXt-272 (2x32d) | 18.34 | 33,020,836 | 4,867.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_cifar100-1834-bbc0c87c.npz.log)) |
| SE-ResNet-20 | 28.54 | 280,697 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_cifar100-2854-1240e42f.npz.log)) |
| SE-ResNet-56 | 22.94 | 868,739 | 127.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_cifar100-2294-ab7e5443.npz.log)) |
| SE-ResNet-110 | 20.86 | 1,750,802 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_cifar100-2086-298d298e.npz.log)) |
| SE-ResNet-164(BN) | 19.95 | 1,929,388 | 256.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_cifar100-1995-cdac82fd.npz.log)) |
| SE-ResNet-272(BN) | 19.07 | 3,176,956 | 422.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_cifar100-1907-a83ac8d6.npz.log)) |
| SE-ResNet-542(BN) | 18.87 | 6,295,876 | 838.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_cifar100-1887-dac530d6.npz.log)) |
| SE-PreResNet-20 | 28.31 | 280,409 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_cifar100-2831-e5480418.npz.log)) |
| SE-PreResNet-56 | 23.05 | 868,451 | 127.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_cifar100-2305-1138b500.npz.log)) |
| SE-PreResNet-110 | 22.61 | 1,750,514 | 255.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_cifar100-2261-b525d8b1.npz.log)) |
| SE-PreResNet-164(BN) | 20.05 | 1,928,012 | 256.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_cifar100-2005-baf00211.npz.log)) |
| SE-PreResNet-272(BN) | 19.13 | 3,175,580 | 422.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_cifar100-1913-d37b7af2.npz.log)) |
| SE-PreResNet-542(BN) | 19.45 | 6,294,500 | 837.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_cifar100-1945-aadac5fb.npz.log)) |
| PyramidNet-110 (a=48) | 20.95 | 1,778,556 | 408.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.186/pyramidnet110_a48_cifar100-2095-b74f12c8.npz.log)) |
| PyramidNet-110 (a=84) | 18.87 | 3,913,536 | 778.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.199/pyramidnet110_a84_cifar100-1887-842b3809.npz.log)) |
| PyramidNet-110 (a=270) | 17.10 | 28,511,307 | 4,730.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.319/pyramidnet110_a270_cifar100-1710-56ae7135.npz.log)) |
| PyramidNet-164 (a=270, BN) | 16.70 | 27,319,071 | 4,608.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet164_a270_bn_cifar100-1670-7614c56c.npz.log)) |
| PyramidNet-200 (a=240, BN) | 16.09 | 26,844,952 | 4,563.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.317/pyramidnet200_a240_bn_cifar100-1684-5dd93682.npz.log)) |
| PyramidNet-236 (a=220, BN) | 16.34 | 27,054,096 | 4,631.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet236_a220_bn_cifar100-1634-fd14728b.npz.log)) |
| PyramidNet-272 (a=200, BN) | 16.19 | 26,288,692 | 4,541.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet272_a200_bn_cifar100-1619-4ba0ea07.npz.log)) |
| DenseNet-40 (k=12) | 24.90 | 622,360 | 210.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.195/densenet40_k12_cifar100-2490-d06839db.npz.log)) |
| DenseNet-BC-40 (k=12) | 28.41 | 188,092 | 74.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.232/densenet40_k12_bc_cifar100-2841-968e5667.npz.log)) |
| DenseNet-BC-40 (k=24) | 22.67 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.221/densenet40_k24_bc_cifar100-2267-f744296d.npz.log)) |
| DenseNet-BC-40 (k=36) | 20.50 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.225/densenet40_k36_bc_cifar100-2050-49b6695f.npz.log)) |
| DenseNet-100 (k=12) | 19.64 | 4,129,600 | 1,353.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.206/densenet100_k12_cifar100-1964-f04f5920.npz.log)) |
| DenseNet-100 (k=24) | 18.08 | 16,236,268 | 5,354.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.318/densenet100_k24_cifar100-1808-47274dd8.npz.log)) |
| DenseNet-BC-100 (k=12) | 21.19 | 800,032 | 298.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.208/densenet100_k12_bc_cifar100-2119-a37ebc2a.npz.log)) |
| DenseNet-BC-250 (k=24) | 17.39 | 15,480,556 | 5,519.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.303/densenet250_k24_bc_cifar100-1739-9100f02a.npz.log)) |
| X-DenseNet-BC-40-2 (k=24) | 23.96 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.227/xdensenet40_2_k24_bc_cifar100-2396-84357bb4.npz.log)) |
| X-DenseNet-BC-40-2 (k=36) | 21.65 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.234/xdensenet40_2_k36_bc_cifar100-2165-9ac51e90.npz.log)) |
| WRN-16-10 | 18.95 | 17,174,324 | 2,414.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.204/wrn16_10_cifar100-1895-d6e85278.npz.log)) |
| WRN-28-10 | 17.88 | 36,536,884 | 5,247.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.320/wrn28_10_cifar100-1788-60387299.npz.log)) |
| WRN-40-8 | 18.03 | 35,794,484 | 5,176.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.321/wrn40_8_cifar100-1803-794aca60.npz.log)) |
| WRN-20-10-1bit | 19.04 | 26,794,920 | 4,022.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_cifar100-1904-1c6f1917.npz.log)) |
| WRN-20-10-32bit | 18.12 | 26,794,920 | 4,022.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_cifar100-1812-346f276f.npz.log)) |
| RoR-3-56 | 25.49 | 768,596 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.229/ror3_56_cifar100-2549-a7903e5f.npz.log)) |
| RoR-3-110 | 23.64 | 1,643,540 | 242.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.236/ror3_110_cifar100-2364-13de922a.npz.log)) |
| RoR-3-164 | 22.34 | 2,518,484 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_cifar100-2234-d5a53210.npz.log)) |
| RiR | 19.23 | 9,527,720 | 1,283.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_cifar100-1923-4bfd2f23.npz.log)) |
| Shake-Shake-ResNet-20-2x16d | 29.22 | 546,932 | 81.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.247/shakeshakeresnet20_2x16d_cifar100-2922-84772a31.npz.log)) |
| Shake-Shake-ResNet-26-2x32d | 18.80 | 2,934,772 | 428.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.222/shakeshakeresnet26_2x32d_cifar100-1880-750a574e.npz.log)) |
| DIA-ResNet-20 | 27.71 | 292,716 | 41.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet20_cifar100-2771-350c5ed4.npz.log)) |
| DIA-ResNet-56 | 24.35 | 876,012 | 129.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet56_cifar100-2435-22e777d2.npz.log)) |
| DIA-ResNet-110 | 22.11 | 1,750,956 | 264.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet110_cifar100-2211-4c6aa3fe.npz.log)) |
| DIA-ResNet-164(BN) | 19.53 | 1,946,132 | 343.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet164bn_cifar100-1953-18aa50ab.npz.log)) |
| DIA-PreResNet-20 | 28.37 | 292,524 | 41.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_cifar100-2837-936a4acc.npz.log)) |
| DIA-PreResNet-56 | 25.05 | 875,820 | 129.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_cifar100-2505-9867b907.npz.log)) |
| DIA-PreResNet-110 | 22.69 | 1,750,764 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_cifar100-2269-0af00d41.npz.log)) |
| DIA-PreResNet-164(BN) | 19.99 | 1,945,236 | 343.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_cifar100-1999-a3835edf.npz.log)) |

### SVHN

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 3.76 | 966,986 | 222.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.270/nin_svhn-0376-2fbe48d0.npz.log)) |
| ResNet-20 | 3.43 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet20_svhn-0343-b6c1dc99.npz.log)) |
| ResNet-56 | 2.75 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet56_svhn-0275-cf18a072.npz.log)) |
| ResNet-110 | 2.45 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet110_svhn-0245-f274056a.npz.log)) |
| ResNet-164(BN) | 2.42 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.267/resnet164bn_svhn-0242-b4c1c66c.npz.log)) |
| ResNet-272(BN) | 2.43 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_svhn-0243-693d5c39.npz.log)) |
| ResNet-542(BN) | 2.34 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_svhn-0234-7421964d.npz.log)) |
| ResNet-1001 | 2.41 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.408/resnet1001_svhn-0241-c8b23d4c.npz.log)) |
| PreResNet-20 | 3.22 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet20_svhn-0322-8e56898f.npz.log)) |
| PreResNet-56 | 2.80 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet56_svhn-0280-f5124073.npz.log)) |
| PreResNet-110 | 2.79 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet110_svhn-0279-8dcd3ae5.npz.log)) |
| PreResNet-164(BN) | 2.58 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet164bn_svhn-0258-69de71f5.npz.log)) |
| PreResNet-272(BN) | 2.34 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_svhn-0234-b2cc8842.npz.log)) |
| PreResNet-542(BN) | 2.36 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_svhn-0236-67f372d8.npz.log)) |
| ResNeXt-29 (32x4d) | 2.80 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.275/resnext29_32x4d_svhn-0280-0a402fab.npz.log)) |
| ResNeXt-29 (16x64d) | 2.68 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.358/resnext29_16x64d_svhn-0268-04ffa539.npz.log)) |
| ResNeXt-272 (1x64d) | 2.35 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_svhn-0235-b12f9d9c.npz.log)) |
| ResNeXt-272 (2x32d) | 2.44 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_svhn-0244-d9432f63.npz.log)) |
| SE-ResNet-20 | 3.23 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_svhn-0323-6c611f0a.npz.log)) |
| SE-ResNet-56 | 2.64 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_svhn-0264-0a017d76.npz.log)) |
| SE-ResNet-110 | 2.35 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_svhn-0235-525399af.npz.log)) |
| SE-ResNet-164(BN) | 2.45 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_svhn-0245-31e8d2be.npz.log)) |
| SE-ResNet-272(BN) | 2.38 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_svhn-0238-2b28cd77.npz.log)) |
| SE-ResNet-542(BN) | 2.26 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_svhn-0226-9571b88b.npz.log)) |
| SE-PreResNet-20 | 3.24 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_svhn-0324-04dafec1.npz.log)) |
| SE-PreResNet-56 | 2.71 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_svhn-0271-150740af.npz.log)) |
| SE-PreResNet-110 | 2.59 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_svhn-0259-eec4c9f3.npz.log)) |
| SE-PreResNet-164(BN) | 2.56 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_svhn-0256-36362d66.npz.log)) |
| SE-PreResNet-272(BN) | 2.49 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_svhn-0249-44b18f81.npz.log)) |
| SE-PreResNet-542(BN) | 2.47 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_svhn-0247-ff5682df.npz.log)) |
| PyramidNet-110 (a=48) | 2.47 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.281/pyramidnet110_a48_svhn-0247-e750bd67.npz.log)) |
| PyramidNet-110 (a=84) | 2.43 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.392/pyramidnet110_a84_svhn-0243-56b06d8f.npz.log)) |
| PyramidNet-110 (a=270) | 2.38 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.393/pyramidnet110_a270_svhn-0238-fdf9f2da.npz.log)) |
| PyramidNet-164 (a=270, BN) | 2.33 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.396/pyramidnet164_a270_bn_svhn-0233-6dcd1882.npz.log)) |
| PyramidNet-200 (a=240, BN) | 2.32 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.397/pyramidnet200_a240_bn_svhn-0232-b5876d02.npz.log)) |
| PyramidNet-236 (a=220, BN) | 2.35 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.398/pyramidnet236_a220_bn_svhn-0235-bb39a3c6.npz.log)) |
| PyramidNet-272 (a=200, BN) | 2.40 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.404/pyramidnet272_a200_bn_svhn-0240-2ace2687.npz.log)) |
| DenseNet-40 (k=12) | 3.05 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.278/densenet40_k12_svhn-0305-8d563cdf.npz.log)) |
| DenseNet-BC-40 (k=12) | 3.20 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.279/densenet40_k12_bc_svhn-0320-52bd7900.npz.log)) |
| DenseNet-BC-40 (k=24) | 2.90 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.280/densenet40_k24_bc_svhn-0290-268af51a.npz.log)) |
| DenseNet-BC-40 (k=36) | 2.60 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.311/densenet40_k36_bc_svhn-0260-47ef4d80.npz.log)) |
| DenseNet-100 (k=12) | 2.60 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.311/densenet100_k12_svhn-0260-c57bbabe.npz.log)) |
| X-DenseNet-BC-40-2 (k=24) | 2.87 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.306/xdensenet40_2_k24_bc_svhn-0287-065f3847.npz.log)) |
| X-DenseNet-BC-40-2 (k=36) | 2.74 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.306/xdensenet40_2_k36_bc_svhn-0274-bf7f7de9.npz.log)) |
| WRN-16-10 | 2.78 | 17,116,634 | 2,414.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.271/wrn16_10_svhn-0278-b87185c8.npz.log)) |
| WRN-28-10 | 2.71 | 36,479,194 | 5,246.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.276/wrn28_10_svhn-0271-59f255be.npz.log)) |
| WRN-40-8 | 2.54 | 35,748,314 | 5,176.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.277/wrn40_8_svhn-0254-8af6aad0.npz.log)) |
| WRN-20-10-1bit | 2.73 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_svhn-0273-4d7bfe0d.npz.log)) |
| WRN-20-10-32bit | 2.59 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_svhn-0259-af3fddd1.npz.log)) |
| RoR-3-56 | 2.69 | 762,746 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.287/ror3_56_svhn-0269-113859bb.npz.log)) |
| RoR-3-110 | 2.57 | 1,637,690 | 242.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.287/ror3_110_svhn-0257-4b8b6963.npz.log)) |
| RoR-3-164 | 2.73 | 2,512,634 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_svhn-0273-1d0a2f12.npz.log)) |
| RiR | 2.68 | 9,492,980 | 1,281.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_svhn-0268-5240bc96.npz.log)) |
| Shake-Shake-ResNet-20-2x16d | 3.17 | 541,082 | 81.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.295/shakeshakeresnet20_2x16d_svhn-0317-261fd59f.npz.log)) |
| Shake-Shake-ResNet-26-2x32d | 2.62 | 2,923,162 | 428.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.295/shakeshakeresnet26_2x32d_svhn-0262-844e1f6d.npz.log)) |
| DIA-ResNet-20 | 3.23 | 286,866 | 41.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet20_svhn-0323-f37bac8b.npz.log)) |
| DIA-ResNet-56 | 2.68 | 870,162 | 129.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet56_svhn-0268-7ea0022b.npz.log)) |
| DIA-ResNet-110 | 2.47 | 1,745,106 | 264.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet110_svhn-0247-515ce8f3.npz.log)) |
| DIA-ResNet-164(BN) | 2.44 | 1,923,002 | 343.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet164bn_svhn-0244-4773b518.npz.log)) |
| DIA-PreResNet-20 | 3.03 | 286,674 | 41.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_svhn-0303-d682b80f.npz.log)) |
| DIA-PreResNet-56 | 2.80 | 869,970 | 129.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_svhn-0280-7a984a63.npz.log)) |
| DIA-PreResNet-110 | 2.42 | 1,744,914 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_svhn-0242-2bab754f.npz.log)) |
| DIA-PreResNet-164(BN) | 2.56 | 1,922,106 | 343.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_svhn-0256-30de9b3b.npz.log)) |

### CUB-200-2011

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-10 | 27.60 | 5,008,392 | 893.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.335/resnet10_cub-2760-e8bdefb0.npz.log)) |
| ResNet-12 | 26.67 | 5,082,376 | 1,125.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.336/resnet12_cub-2667-22b2b216.npz.log)) |
| ResNet-14 | 24.34 | 5,377,800 | 1,357.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.337/resnet14_cub-2434-57f6a73d.npz.log)) |
| ResNet-16 | 23.21 | 6,558,472 | 1,588.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.338/resnet16_cub-2321-5e48b19f.npz.log)) |
| ResNet-18 | 23.33 | 11,279,112 | 1,820.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.344/resnet18_cub-2333-c32998b4.npz.log)) |
| ResNet-26 | 22.61 | 17,549,832 | 2,746.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.345/resnet26_cub-2261-56c8fcc1.npz.log)) |
| SE-ResNet-10 | 27.42 | 5,052,932 | 893.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet10_cub-2742-b8e56acf.npz.log)) |
| SE-ResNet-12 | 25.99 | 5,127,496 | 1,126.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet12_cub-2599-9c0ee8cf.npz.log)) |
| SE-ResNet-14 | 23.68 | 5,425,104 | 1,357.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet14_cub-2368-b58cddb7.npz.log)) |
| SE-ResNet-16 | 23.18 | 6,614,240 | 1,589.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet16_cub-2318-1d8b187c.npz.log)) |
| SE-ResNet-18 | 23.21 | 11,368,192 | 1,820.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet18_cub-2321-7b1d02a7.npz.log)) |
| SE-ResNet-26 | 22.54 | 17,683,452 | 2,747.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet26_cub-2254-5cbf65d2.npz.log)) |
| MobileNet x1.0 | 23.56 | 3,411,976 | 578.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.346/mobilenet_w1_cub-2356-02c2accf.npz.log)) |
| ProxylessNAS Mobile | 21.90 | 3,055,712 | 331.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.347/proxylessnas_mobile_cub-2190-a9c66b1b.npz.log)) |
| NTS-Net | 12.86 | 28,623,333 | 33,361.79M | From [yangze0930/NTS-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.334/ntsnet_cub-1286-4d759524.npz.log)) |

### Pascal VOC20102

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 96.57 | 76.26 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_voc-7626-f90c0db9.npz.log)) |
| DeepLabv3 | ResNet(D)-101b | 96.57 | 75.66 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_voc-7566-6a4f805f.npz.log)) |
| DeepLabv3 | ResNet(D)-152b | 97.25 | 78.06 | 74,398,421 | 59,894.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd152b_voc-7806-1c3089b5.npz.log)) |
| FCN-8s(d) | ResNet(D)-101b | 97.80 | 80.40 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_voc-8040-3568dc41.npz.log)) |

### ADE20K

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-50b | 68.95 | 27.46 | 46,782,550 | 162,595.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd50b_ade20k-2746-7b7ce568.npz.log)) |
| PSPNet | ResNet(D)-101b | 75.14 | 32.86 | 65,774,678 | 231,008.79M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_ade20k-3286-c5e619c4.npz.log)) |
| DeepLabv3 | ResNet(D)-50b | 74.63 | 31.96 | 39,795,798 | 32,756.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd50b_ade20k-3196-00903dce.npz.log)) |
| DeepLabv3 | ResNet(D)-101b | 77.81 | 35.17 | 58,787,926 | 47,651.23M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_ade20k-3517-46828740.npz.log)) |
| FCN-8s(d) | ResNet(D)-50b | 76.92 | 33.39 | 33,146,966 | 128,387.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd50b_ade20k-3339-1d03bc38.npz.log)) |
| FCN-8s(d) | ResNet(D)-101b | 79.01 | 35.88 | 52,139,094 | 196,800.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_ade20k-3588-ff385e19.npz.log)) |

### Cityscapes

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 92.72 | 57.57 | 65,707,475 | 230,767.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_cityscapes-5757-2e2315d4.npz.log)) |
| ICNet | ResNet(D)-50b | 95.24 | 60.78 | 47,489,184 | 14,253.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.457/icnet_resnetd50b_cityscapes-6078-04f581dc.npz.log)) |
| SINet | - | 93.71 | 60.84 | 119,418 | 1,419.90M | From [clovaai/c3_sinet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.437/sinet_cityscapes-6084-c0a4e992.npz.log)) |
| Fast-SCNN | - | 95.11 | 65.95 | 1,138,051 | 3493.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.474/fastscnn_cityscapes-6595-6dca4260.npz.log)) |
| DANet | ResNet(D)-50b | 95.91 | 67.99 | 47,586,427 | 180,397.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd50b_cityscapes-6799-dcef11be.npz.log)) |
| DANet | ResNet(D)-101b | 96.03 | 68.10 | 66,578,555 | 248,811.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd101b_cityscapes-6810-a6593e21.npz.log)) |

### COCO Semantic Segmentation

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 88.99 | 54.67 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_coco-5467-69033558.npz.log)) |
| DeepLabv3 | ResNet(D)-101b | 90.10 | 59.06 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_coco-5906-2811b3cd.npz.log)) |
| DeepLabv3 | ResNet(D)-152b | 90.52 | 61.07 | 74,398,421 | 275,087.91M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd152b_coco-6107-80ddcd96.npz.log)) |
| FCN-8s(d) | ResNet(D)-101b | 91.44 | 60.11 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_coco-6011-4a469997.npz.log)) |

### CelebAMask-HQ

| Model | Extractor | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | --- |
| BiSeNet | ResNet-18 | 13,300,416 | - | From [zllrunning/face...Torch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.462/bisenet_resnet18_celebamaskhq-0000-c3bd2251.npz.log)) |

### COCO Keypoints Detection

| Model | Extractor | OKS AP, % | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | --- |
| AlphaPose | Fast-SE-ResNet-101b | 74.15/91.59/80.68 | 59,569,873 | 9,553.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.454/alphapose_fastseresnet101b_coco-7415-c1aee8e0.npz.log)) |
| SimplePose | ResNet-18 | 66.31/89.20/73.41 | 15,376,721 | 1,799.25M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet18_coco-6631-e267629f.npz.log)) |
| SimplePose | ResNet-50b | 71.02/91.23/78.57 | 33,999,697 | 4,041.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet50b_coco-7102-78b005c8.npz.log)) |
| SimplePose | ResNet-101b | 72.44/92.18/79.76 | 52,991,825 | 7,685.04M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet101b_coco-7244-59f85623.npz.log)) |
| SimplePose | ResNet-152b | 72.53/92.14/79.61 | 68,635,473 | 11,332.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet152b_coco-7253-6228ce42.npz.log)) |
| SimplePose | ResNet(A)-50b | 71.70/91.31/78.66 | 34,018,929 | 4,278.56M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta50b_coco-7170-e45c6525.npz.log)) |
| SimplePose | ResNet(A)-101b | 72.97/92.24/80.81 | 53,011,057 | 7,922.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta101b_coco-7297-80050053.npz.log)) |
| SimplePose | ResNet(A)-152b | 73.44/92.27/80.72 | 68,654,705 | 11,570.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta152b_coco-7344-ac76d0a9.npz.log)) |
| SimplePose(Mobile) | ResNet-18 | 66.25/89.17/74.32 | 12,858,208 | 1,960.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet18_coco-6625-a5201083.npz.log)) |
| SimplePose(Mobile) | ResNet-50b | 71.10/91.28/78.67 | 25,582,944 | 4,221.30M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet50b_coco-7110-6d17c89b.npz.log)) |
| SimplePose(Mobile) | 1.0 MobileNet-224 | 64.10/88.06/71.23 | 5,019,744 | 751.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenet_w1_coco-6410-14efcbba.npz.log)) |
| SimplePose(Mobile) | 1.0 MobileNetV2b-224 | 63.74/88.12/71.06 | 4,102,176 | 495.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv2b_w1_coco-6374-73b90839.npz.log)) |
| SimplePose(Mobile) | MobileNetV3 Small 224/1.0 | 54.34/83.67/59.35 | 2,625,088 | 236.51M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_small_w1_coco-5434-cc5169a3.npz.log)) |
| SimplePose(Mobile) | MobileNetV3 Large 224/1.0 | 63.67/88.91/70.82 | 4,768,336 | 403.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_large_w1_coco-6367-b93dbd09.npz.log)) |
| Lightweight OpenPose 2D | MobileNet | 39.99/65.95/40.70 | 4,091,698 | 8,948.96M | From [Daniil-Osokin/lighw...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose2d_mobilenet_cmupan_coco-3999-0a2829dc.npz.log)) |
| Lightweight OpenPose 3D | MobileNet | 39.99/65.95/40.70 | 5,085,983 | 11,049.43M | From [Daniil-Osokin/li...3d...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose3d_mobilenet_cmupan_coco-3999-ef1e8e13.npz.log)) |
| IBPPose | - | 64.87/83.62/70.12 | 95,827,784 | 57,195.91M | From [jialee93/Improved...Parts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.459/ibppose_coco-6487-70158be1.npz.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[ShichenLiu/CondenseNet]: https://github.com/ShichenLiu/CondenseNet
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[clavichord93/FD-MobileNet]: https://github.com/clavichord93/FD-MobileNet
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[dyhan0920/Pyramid...PyTorch]: https://github.com/dyhan0920/PyramidNet-PyTorch
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[szagoruyko/diracnets]: https://github.com/szagoruyko/diracnets
[szagoruyko/functional-zoo]: https://github.com/szagoruyko/functional-zoo
[fyu/drn]: https://github.com/fyu/drn
[quark0/darts]: https://github.com/quark0/darts
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[kevin-ssy/FishNet]: https://github.com/kevin-ssy/FishNet
[ucbdrive/dla]: https://github.com/ucbdrive/dla
[sacmehta/ESPNetv2]: https://github.com/sacmehta/ESPNetv2
[sacmehta/EdgeNets]: https://github.com/sacmehta/EdgeNets
[jhjacobsen/pytorch-i-revnet]: https://github.com/jhjacobsen/pytorch-i-revnet
[wielandbrendel/bag...models]: https://github.com/wielandbrendel/bag-of-local-features-models
[MIT-HAN-LAB/ProxylessNAS]: https://github.com/MIT-HAN-LAB/ProxylessNAS
[yangze0930/NTS-Net]: https://github.com/yangze0930/NTS-Net
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models
[HRNet/HRNet...ation]: https://github.com/HRNet/HRNet-Image-Classification
[stigma0617/VoVNet.pytorch]: https://github.com/stigma0617/VoVNet.pytorch
[PingoLH/Pytorch-HarDNet]: https://github.com/PingoLH/Pytorch-HarDNet
[clovaai/c3_sinet]: https://github.com/clovaai/c3_sinet
[Daniil-Osokin/lighw...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
[Daniil-Osokin/li...3d...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
[jialee93/Improved...Parts]: https://github.com/jialee93/Improved-Body-Parts
[zllrunning/face...Torch]: https://github.com/zllrunning/face-parsing.PyTorch
[MCG-NKU/SCNet]: https://github.com/MCG-NKU/SCNet