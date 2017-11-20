from keras import backend as K
import numpy as np
from keras.layers import Convolution2D

def GetScalingFactor(bitDepth, maxValue):
    dynamicRange = 2 ** bitDepth
    scalingFactor = dynamicRange / (2 * maxValue)
    return scalingFactor


class INQLayer(Convolution2D):
    """2D convolution layer with Incremental Network Quantisation Functionality

    This layer implements the 'Incremental Network Quantisation' paper:
        Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights

    Currently this layer implements evenly spaced weights, not POW 2 weights as described in the paper


    NOTE: The INQ functionality is implemented using a 'on batch end' function,
    which must be called at the end of every batch (ideally using a callback).  Currently this functionality
    is implemented using Numpy.  In practice this incurs a negligible performance penalty,
    as this function uses far fewer operations than the base convolution operation.

    # Arguments
        numIterations: Number of epochs to perform INQ over (before quantising all weights)
        maxValue: Maximum number space value
        bitDepth: Number of bits to quantise weights to

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """


    def __init__(self, start_epoch, numIterations, maxValue, bitDepth, **kwargs):
        super().__init__(**kwargs)
        self.maxValue = maxValue
        self.bitDepth = bitDepth
        self.start_epoch = start_epoch
        self.numIterations = numIterations


    def build(self, input_shape):
        super().build(input_shape=input_shape)

        self.cur_epoch = 0

        # This value converts from floating point to our fixed point number system
        self.scalingFactor = GetScalingFactor(bitDepth=self.bitDepth, maxValue=self.maxValue)

        # Get the last iteration weights
        weights = K.get_value(self.weights[0])
        self.lastIterationWeights = weights.copy()

        # Weight quantisation mask.  Initialised to 0, no weights quantised
        self.quantisedMask = np.zeros(shape=weights.shape, dtype='uint8')


    def get_config(self):
        config = {'maxValue': self.maxValue,
                  'bitDepth': self.bitDepth,
                  'numIterations': self.numIterations,
                  'start_epoch': self.start_epoch}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def on_batch_end(self, batch):

        # Only update the non-quantised weights
        weights = K.get_value(self.weights[0])
        quantised_weights = self.lastIterationWeights * self.quantisedMask
        non_quantised_weights = (1 - self.quantisedMask) * weights

        weights_new = quantised_weights + non_quantised_weights
        K.set_value(x=self.weights[0], value=weights_new)

        self.lastIterationWeights = weights_new


    def on_epoch_begin(self, epoch):

        # On the last epoch, quantise everything
        if self.cur_epoch >= (self.numIterations + self.start_epoch):
            self.quantisedMask = np.ones(shape=self.lastIterationWeights.shape)

        # Update the quantisation mask
        elif self.cur_epoch >= self.start_epoch:

            # First figure out the quantisation threshold
            # This is the value that partitions the remaining non-quantised weights into two
            weights = K.get_value(self.weights[0])

            non_quantised_weights = (1. - self.quantisedMask) * weights
            non_quantised_weights = non_quantised_weights[non_quantised_weights != 0]
            non_quantised_weights_1D = np.abs(np.ravel(non_quantised_weights))
            non_quantised_weights_1D = np.sort(non_quantised_weights_1D)

            # Get the current threshold value for quantisation
            entry = int(len(non_quantised_weights_1D) * 0.5)
            entry = min(len(non_quantised_weights_1D) - 1, entry)
            quantisation_threshold = non_quantised_weights_1D[entry]

            # Now update the mask based on weight magnitude
            quantisedThisIteration = np.greater(np.abs(weights), quantisation_threshold)
            self.quantisedMask = np.logical_or(self.quantisedMask, quantisedThisIteration)

        self.cur_epoch += 1

        numQuantisedWeights = 100. * np.sum(self.quantisedMask) / self.quantisedMask.size
        print('Percentage of weights quantised is: %f' % numQuantisedWeights)

        # Quantise the selected weights
        weights = K.get_value(self.weights[0])
        quantised_weights = self.quantisedMask * weights

        quantised_weights = np.around(quantised_weights * self.scalingFactor)
        quantised_weights = quantised_weights / self.scalingFactor

        # Combine the newly quantised weights with the rest and save
        non_quantised_weights = (1 - self.quantisedMask) * weights
        weights_new = quantised_weights + non_quantised_weights
        K.set_value(x=self.weights[0], value=weights_new)

        self.lastIterationWeights = weights_new
