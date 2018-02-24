import keras.backend as K
from keras.layers import Convolution2D
from keras.engine import InputSpec
import numpy as np
from keras import initializers


class DNSConv2D(Convolution2D):
    """2D convolution layer with Dynamic Network Surgery Functionality

    This layer implements the 'Dynamic Network Surgery' paper:
        Dynamic Network Surgery for Efficient DNNs

    Two things weren't described in the paper, namely how the authors calculated the weight switching probability threshold
    and how to judge if a weight is significant or not.  The implementations chosen below were obtained from the author's
    C++ source code.

    The weight significance function basically considers how large a weight is.  Namely how many standard deviations
    it is away from the weight mean.

    NOTE: There are two methods of engaging DNS: Start epoch or by calling the start_DNS() method

    # Arguments
        start_epoch: Training epoch at which to start applying DNS
        gamma: Decay constant for the mask switching probability calculation
        crate: Standard deviations away from the mean at which to start pruning weights.
            Taken from the author's C++ code.  This variable controls the amount of weights that are pruned.
        power: DNS filter update probability decay exponent.  Larger values will cause DNS to finish sooner
        print_batch: Print DNS statistics at the end of every batch
        print_epoch: Print DNS statistics at the end of every epoch

    # Input shape
        4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 start_epoch=-1,     # Epoch at which to begin DNS.  Can also start DNS by calling method below
                 gamma=0.0001,       # Speed at which mask switch probability drops
                 crate=3.,           # Some sort of outlier/significancy threshold (nominally 4 in their code)
                 power=-1,           # Probability decay exponent
                 print_batch=False,  # Print DNS statistics at the end of every batch
                 print_epoch=True,   # Print DNS statistics at the end of every epoch
                 **kwargs):

        super().__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)

        self.start_epoch = start_epoch
        self.gamma = gamma
        self.crate = crate
        self.power = power
        self.print_batch = print_batch
        self.print_epoch = print_epoch


    def build(self, input_shape):
        # Call the build function of the base class (in this case, convolution)
        super().build(input_shape)  # Be sure to call this somewhere!

        # Current DNS iteration
        self.cur_iter = self.add_weight(shape=None,
                                        initializer=initializers.get('zeros'),
                                        trainable=False,
                                        name='current_DNS_iteration',
                                        )#dtype='int32')
        K.set_value(x=self.cur_iter, value=-1)

        # Mask variable.  Initialise to 1 so that every weight is unmasked during training before DNS starts
        self.T = self.add_weight(shape=self.kernel.shape,
                                 initializer=initializers.get('ones'),
                                 name='T',
                                 trainable=False)


    def get_config(self):
        config = {'start_epoch': self.start_epoch,
                  'gamma': self.gamma,
                  'crate': self.crate,
                  'power': self.power,
                  'print_batch': self.print_batch,
                  'print_epoch': self.print_epoch,
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def call(self, inputs):
        # Perform the base convolution
        outputs = K.conv2d(
            inputs,
            self.T * self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        # Add the bias
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        # Add the appropriate activation, if specified
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Update the current DNS iteration
        cur_iter_new = K.switch(self.cur_iter >= 0., self.cur_iter + 1., self.cur_iter)
        self.add_update(K.update(self.cur_iter, cur_iter_new), )

        # Update the weight mask
        self.add_update(K.update(self.T, DNS_update(cur_iter=self.cur_iter,
                                                    gamma=self.gamma,
                                                    crate=self.crate,
                                                    power=self.power,
                                                    kernel=self.kernel,
                                                    T=self.T,
                                                    learning_phase=K.learning_phase())),
                        # cur_iter_new > 0, # having some trouble with making the update conditional here
                        )

        return outputs


    def on_batch_end(self, batch):
        if self.print_batch:
            T = K.get_value(x=self.T)
            cur_iter = K.get_value(self.cur_iter)

            if cur_iter != -1.:
                print('Percentage of weights unmasked at the end of this batch is: %f' % (100.0 * np.sum(T) / T.size))
                print('Number of weights unmasked at the end of this batch is: %d, which is iteration %d of DNS' % (np.sum(T), cur_iter))


    def start_DNS(self):
        K.set_value(x=self.cur_iter, value=0.)


    def on_epoch_begin(self, epoch):
        cur_iter = K.get_value(x=self.cur_iter)

        if epoch >= self.start_epoch and cur_iter == -1.:
            self.start_DNS()


    def on_epoch_end(self, epoch):
        T = K.get_value(x=self.T)
        cur_iter = K.get_value(self.cur_iter)

        if cur_iter != -1. and self.print_epoch:
            print('Percentage of weights unmasked at the end of this epoch is: %f' % (100.0 * np.sum(T) / T.size))
            print('Number of weights unmasked at the end of this epoch is: %d, which is iteration %d of DNS' % (np.sum(T), cur_iter))


def DNS_update(cur_iter, gamma, crate, power, kernel, T, learning_phase):
    # The paper isn't too clear on this, so this has been dug out of their C++ source code
    probThreshold = (1 + gamma * cur_iter) ** power

    # Determine which filters shall be updated this iteration
    random_number = K.random_uniform(shape=(1, 1, 1, int(T.shape[-1])))
    random_number = K.cast(random_number < probThreshold, dtype='float32')

    # Based on the mean & standard deviation of the weights, determine a weight significancy threshold
    mu_vec = K.mean(x=kernel, axis=(0, 1, 2), keepdims=True)
    std_vec = K.std(x=kernel, axis=(0, 1, 2), keepdims=True)
    threshold_vec = mu_vec + crate * std_vec  # weights this many std. deviations from the mean are 'significant'

    # Incorporate hysteresis into the threshold
    alpha_vec = 0.9 * threshold_vec
    beta_vec = 1.1 * threshold_vec

    # Update the significant weight mask by applying the threshold to the unmasked weights
    abs_kernel = K.abs(x=kernel)
    new_T = T - K.cast(abs_kernel < alpha_vec, dtype='float32') * random_number
    new_T = new_T + K.cast(abs_kernel > beta_vec, dtype='float32') * random_number
    new_T = K.clip(x=new_T, min_value=0., max_value=1.)

    # Only apply DNS when training and when activated via the current iteration variable
    new_T = K.switch(cur_iter >= 0., new_T, T)
    new_T = K.switch(learning_phase, new_T, T)

    return new_T