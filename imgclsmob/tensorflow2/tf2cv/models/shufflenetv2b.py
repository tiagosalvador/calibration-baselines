"""
    ShuffleNet V2 for ImageNet-1K, implemented in TensorFlow. The alternative version.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['ShuffleNetV2b', 'shufflenetv2b_wd2', 'shufflenetv2b_w1', 'shufflenetv2b_w3d2', 'shufflenetv2b_w2']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, ChannelShuffle, ChannelShuffle2, SEBlock, MaxPool2d,\
    SimpleSequential, get_channel_axis, flatten


class ShuffleUnit(nn.Layer):
    """
    ShuffleNetV2(b) unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether do downsample.
    use_se : bool
        Whether to use SE block.
    use_residual : bool
        Whether to use residual connection.
    shuffle_group_first : bool
        Whether to use channel shuffle in group first mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 shuffle_group_first,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.downsample = downsample
        self.use_se = use_se
        self.use_residual = use_residual
        mid_channels = out_channels // 2
        in_channels2 = in_channels // 2
        assert (in_channels % 2 == 0)

        y2_in_channels = (in_channels if downsample else in_channels2)
        y2_out_channels = out_channels - y2_in_channels

        self.conv1 = conv1x1_block(
            in_channels=y2_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.dconv = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=(2 if self.downsample else 1),
            activation=None,
            data_format=data_format,
            name="dconv")
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=y2_out_channels,
            data_format=data_format,
            name="conv2")
        if self.use_se:
            self.se = SEBlock(
                channels=y2_out_channels,
                data_format=data_format,
                name="se")
        if downsample:
            self.shortcut_dconv = dwconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2,
                activation=None,
                data_format=data_format,
                name="shortcut_dconv")
            self.shortcut_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                data_format=data_format,
                name="shortcut_conv")

        if shuffle_group_first:
            self.c_shuffle = ChannelShuffle(
                channels=out_channels,
                groups=2,
                data_format=data_format,
                name="c_shuffle")
        else:
            self.c_shuffle = ChannelShuffle2(
                channels=out_channels,
                groups=2,
                data_format=data_format,
                name="c_shuffle")

    def call(self, x, training=None):
        if self.downsample:
            y1 = self.shortcut_dconv(x, training=training)
            y1 = self.shortcut_conv(y1, training=training)
            x2 = x
        else:
            y1, x2 = tf.split(x, num_or_size_splits=2, axis=get_channel_axis(self.data_format))
        y2 = self.conv1(x2, training=training)
        y2 = self.dconv(y2, training=training)
        y2 = self.conv2(y2, training=training)
        if self.use_se:
            y2 = self.se(y2)
        if self.use_residual and not self.downsample:
            y2 = y2 + x2
        x = tf.concat([y1, y2], axis=get_channel_axis(self.data_format))
        x = self.c_shuffle(x)
        return x


class ShuffleInitBlock(nn.Layer):
    """
    ShuffleNetV2(b) specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleInitBlock, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            ceil_mode=False,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class ShuffleNetV2b(tf.keras.Model):
    """
    ShuffleNetV2(b) model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    use_se : bool, default False
        Whether to use SE block.
    use_residual : bool, default False
        Whether to use residual connections.
    shuffle_group_first : bool, default True
        Whether to use channel shuffle in group first mode.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 use_se=False,
                 use_residual=False,
                 shuffle_group_first=True,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleNetV2b, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(ShuffleInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                stage.add(ShuffleUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample=downsample,
                    use_se=use_se,
                    use_residual=use_residual,
                    shuffle_group_first=shuffle_group_first,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_shufflenetv2b(width_scale,
                      shuffle_group_first=True,
                      model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".tensorflow", "models"),
                      **kwargs):
    """
    Create ShuffleNetV2(b) model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    shuffle_group_first : bool, default True
        Whether to use channel shuffle in group first mode.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 24
    final_block_channels = 1024
    layers = [4, 8, 4]
    channels_per_layers = [116, 232, 464]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)

    net = ShuffleNetV2b(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        shuffle_group_first=shuffle_group_first,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def shufflenetv2b_wd2(**kwargs):
    """
    ShuffleNetV2(b) 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2b(
        width_scale=(12.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_wd2",
        **kwargs)


def shufflenetv2b_w1(**kwargs):
    """
    ShuffleNetV2(b) 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2b(
        width_scale=1.0,
        shuffle_group_first=True,
        model_name="shufflenetv2b_w1",
        **kwargs)


def shufflenetv2b_w3d2(**kwargs):
    """
    ShuffleNetV2(b) 1.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2b(
        width_scale=(44.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_w3d2",
        **kwargs)


def shufflenetv2b_w2(**kwargs):
    """
    ShuffleNetV2(b) 2x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2b(
        width_scale=(61.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_w2",
        **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        shufflenetv2b_wd2,
        shufflenetv2b_w1,
        shufflenetv2b_w3d2,
        shufflenetv2b_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shufflenetv2b_wd2 or weight_count == 1366792)
        assert (model != shufflenetv2b_w1 or weight_count == 2279760)
        assert (model != shufflenetv2b_w3d2 or weight_count == 4410194)
        assert (model != shufflenetv2b_w2 or weight_count == 7611290)


if __name__ == "__main__":
    _test()
