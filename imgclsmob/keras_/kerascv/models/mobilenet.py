"""
    MobileNet & FD-MobileNet for ImageNet-1K, implemented in Keras.
    Original papers:
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
       https://arxiv.org/abs/1704.04861.
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.
"""

__all__ = ['mobilenet', 'mobilenet_w1', 'mobilenet_w3d4', 'mobilenet_wd2', 'mobilenet_wd4', 'fdmobilenet_w1',
           'fdmobilenet_w3d4', 'fdmobilenet_wd2', 'fdmobilenet_wd4']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, is_channels_first, flatten


def dws_conv_block(x,
                   in_channels,
                   out_channels,
                   strides,
                   name="dws_conv_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers. It is used as
    a MobileNet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'dws_conv_block'
        Block name.

    Returns:
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = dwconv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=in_channels,
        strides=strides,
        name=name + "/dw_conv")
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        name=name + "/pw_conv")
    return x


def mobilenet(channels,
              first_stage_stride,
              in_channels=3,
              in_size=(224, 224),
              classes=1000):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861. Also this class implements FD-MobileNet from 'FD-MobileNet: Improved MobileNet
    with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, in_size[0], in_size[1]) if is_channels_first() else\
        (in_size[0], in_size[1], in_channels)
    input = nn.Input(shape=input_shape)

    init_block_channels = channels[0][0]
    x = conv3x3_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        strides=2,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels[1:]):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
            x = dws_conv_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    # x = nn.Flatten()(x)
    x = flatten(x)
    x = nn.Dense(
        units=classes,
        input_dim=in_channels,
        name="output")(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_mobilenet(version,
                  width_scale,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".keras", "models"),
                  **kwargs):
    """
    Create MobileNet or FD-MobileNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('orig' or 'fd').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if version == 'orig':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
        first_stage_stride = False
    elif version == 'fd':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 1024]]
        first_stage_stride = True
    else:
        raise ValueError("Unsupported MobileNet version {}".format(version))

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = mobilenet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def mobilenet_w1(**kwargs):
    """
    1.0 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=1.0, model_name="mobilenet_w1", **kwargs)


def mobilenet_w3d4(**kwargs):
    """
    0.75 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.75, model_name="mobilenet_w3d4", **kwargs)


def mobilenet_wd2(**kwargs):
    """
    0.5 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.5, model_name="mobilenet_wd2", **kwargs)


def mobilenet_wd4(**kwargs):
    """
    0.25 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.25, model_name="mobilenet_wd4", **kwargs)


def fdmobilenet_w1(**kwargs):
    """
    FD-MobileNet 1.0x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=1.0, model_name="fdmobilenet_w1", **kwargs)


def fdmobilenet_w3d4(**kwargs):
    """
    FD-MobileNet 0.75x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.75, model_name="fdmobilenet_w3d4", **kwargs)


def fdmobilenet_wd2(**kwargs):
    """
    FD-MobileNet 0.5x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.5, model_name="fdmobilenet_wd2", **kwargs)


def fdmobilenet_wd4(**kwargs):
    """
    FD-MobileNet 0.25x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.25, model_name="fdmobilenet_wd4", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        mobilenet_w1,
        mobilenet_w3d4,
        mobilenet_wd2,
        mobilenet_wd4,
        fdmobilenet_w1,
        fdmobilenet_w3d4,
        fdmobilenet_wd2,
        fdmobilenet_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenet_w1 or weight_count == 4231976)
        assert (model != mobilenet_w3d4 or weight_count == 2585560)
        assert (model != mobilenet_wd2 or weight_count == 1331592)
        assert (model != mobilenet_wd4 or weight_count == 470072)
        assert (model != fdmobilenet_w1 or weight_count == 2901288)
        assert (model != fdmobilenet_w3d4 or weight_count == 1833304)
        assert (model != fdmobilenet_wd2 or weight_count == 993928)
        assert (model != fdmobilenet_wd4 or weight_count == 383160)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
