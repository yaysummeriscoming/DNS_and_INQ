from keras.layers import Activation, BatchNormalization
from CustomLayers.INQLayer import INQLayer

def INQConvReluLayer(input,
                     nb_filters,
                     border,
                     kernel_size,
                     stride,
                     start_epoch,
                     num_iterations,
                     bit_depth=5,
                     max_value=4.0,
                     use_bias=True,
                     data_format='channels_last',
                     use_BN=True,
                     use_activation=True):

    output = INQLayer(filters=nb_filters,
                      kernel_size=kernel_size,
                      use_bias=use_bias,
                      padding=border,
                      strides=stride,
                      data_format=data_format,
                      start_epoch=start_epoch,
                      numIterations=num_iterations,
                      maxValue=max_value,
                      bitDepth=bit_depth
                      )(input)

    if use_BN:
        output = BatchNormalization()(output)

    if use_activation:
        output = Activation('relu')(output)

    return output



