from keras.callbacks import Callback

from CustomLayers.DNSConvLayer import DNSConv2D
from CustomLayers.INQLayer import INQLayer


# This file holds a dictionary of all custom layers, for use when loading a Keras model
customLayersDictionary = {
                          "DNSConv2D": DNSConv2D,
                          "INQLayer" : INQLayer,
                          }

class CustomLayerUpdate(Callback):

    def on_batch_begin(self, batch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, iteration=batch, fn_name='on_batch_begin')

    def on_batch_end(self, batch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, iteration=batch, fn_name='on_batch_end')

    def on_epoch_begin(self, epoch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, iteration=epoch, fn_name='on_epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, iteration=epoch, fn_name='on_epoch_end')


# Call the desired class method if it exists
def CallMethodName(object, iteration, fn_name):
    fn = getattr(object, fn_name, None)
    if callable(fn):
        fn(iteration)

# Callbacks to implement custom layer specific code at the end of each training batch
customLayerCallbacks = [CustomLayerUpdate()]
