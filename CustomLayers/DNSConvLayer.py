import keras.backend as K
from keras.layers import Convolution2D
from keras.engine import InputSpec
import numpy as np
from random import uniform

class DNSConv2D(Convolution2D):
    """2D convolution layer with Dynamic Network Surgery Functionality

    This layer implements the 'Dynamic Network Surgery' paper:
        Dynamic Network Surgery for Efficient DNNs

    Two things weren't described in the paper, namely how the authors calculated the weight switching probability threshold
    and how to judge if a weight is significant or not.  The implementations chosen below were obtained from the author's
    C++ source code.

    The weight significance function basically considers how large a weight is.  Namely how many standard deviations
    it is away from the weight mean.

    NOTE: The pruning functionality is implemented using a 'on batch end' function,
    which must be called at the end of every batch (ideally using a callback).  Currently this functionality
    is implemented using Numpy.  In practice this incurs a negligible performance penalty,
    as this function uses far fewer operations than the base convolution operation.

    # Arguments
        start_epoch: Training epoch at which to start applying DNS
        gamma: Decay constant for the mask switching probability calculation
        crate: Standard deviations away from the mean at which to start pruning weights.
            Taken from the author's C++ code.  This variable controls the amount of weights that are pruned.

    # Input shape
        4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 start_epoch,
                 gamma=0.0001,
                 crate=3.,
                 **kwargs):

        super().__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)

        assert self.data_format == 'channels_last'
        self.start_epoch = start_epoch

        # Probability of updating T:
        self.gamma = gamma  # Speed at which mask switch probability drops
        self.crate = crate  # Some sort of outlier/significancy threshold (nominally 4 in their code)

        self.power = -1  # Probability decay exponent
        self.curIter = -1



    def build(self, input_shape):
        # Call the build function of the base class (in this case, convolution)
        super().build(input_shape)  # Be sure to call this somewhere!

        self.unmasked_weights = K.get_value(self.weights[0]).copy()
        self.lastIterationWeights = K.get_value(self.weights[0]).copy()

        # Mask variable.  Initialise to 1 so that every weight is unmasked during training before DNS starts
        self.T = np.ones(shape=self.unmasked_weights.shape)



    def get_config(self):
        config = {'start_epoch': self.start_epoch,
                  'gamma': self.gamma,
                  'crate': self.crate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def on_batch_end(self, batch):
        # First update the unmasked weights
        weightsUpdate = K.get_value(self.weights[0]) - self.lastIterationWeights
        self.unmasked_weights += weightsUpdate

        if self.curIter >= 0:
            # The paper isn't too clear on this, so this has been dug out of their C++ source code
            probThreshold = (1 + self.gamma * self.curIter) ** self.power

            # Roll the output channels dimension to the front to allow for easy looping
            # Weight arrangement is (channels last notation:
            # (kernel_size, kernel_size, num_input_channels, num_output_channels)
            weights_reshuffled = np.swapaxes(self.unmasked_weights, 0, -1)
            t_reshuffled = np.swapaxes(self.T, 0, -1)

            # Loop over each output mask & filter in the layer
            for curT, curFilter in zip(t_reshuffled, weights_reshuffled):

                # Now generate a random number and see if we're going to update this filter
                randomNumber = uniform(0.0, 1.0)
                if randomNumber < probThreshold:
                    mu = np.mean(curFilter)              # mean
                    std = np.std(curFilter)              # standard deviation
                    threshold = mu + self.crate * std    # weights this many std. deviations from the mean are 'significant'
                    alpha = 0.9 * threshold
                    beta  = 1.1 * threshold

                    # Mask & unmask weights according to magnitude.
                    # Note that there is a deadband in which weights aren't touched
                    curT[np.abs(curFilter) < alpha] = 0
                    curT[np.abs(curFilter) > beta] = 1

            self.curIter += 1

        # Apply the output mask & save the resulting masked weights ready for the next batch
        output_weights = self.T * self.unmasked_weights
        self.lastIterationWeights = output_weights.copy()
        K.set_value(self.weights[0], output_weights)


    def on_epoch_end(self, epoch):
        if epoch >= self.start_epoch and self.curIter == -1:
            self.curIter = 0

        print('Percentage of weights unmasked at the end of this batch is: %f\n' % (100.0 * np.sum(self.T) / self.T.size))
        print('Number of weights unmasked at the end of this batch is: %d, which is iteration %d of DNS\n' % (np.sum(self.T), self.curIter))
