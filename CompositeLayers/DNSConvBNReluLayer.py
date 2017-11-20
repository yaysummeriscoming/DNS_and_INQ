from keras.layers import Activation, BatchNormalization
from CustomLayers.DNSConvLayer import DNSConv2D


def DNSConvBNReluLayer(input,
                       nb_filters,
                       border,
                       kernel_size,
                       stride,
                       start_epoch,
                       gamma,
                       crate,
                       use_bias=True,
                       use_BN=True,
                       use_activation=True,
                       data_format='channels_last'):

    output = DNSConv2D(filters=nb_filters,
                       kernel_size=kernel_size,
                       start_epoch=start_epoch,
                       gamma=gamma,
                       crate=crate,
                       use_bias=use_bias,
                       padding=border,
                       strides=stride,
                       data_format=data_format
                       )(input)

    if use_BN:
        output = BatchNormalization()(output)

    if use_activation:
        output = Activation('relu')(output)

    return output