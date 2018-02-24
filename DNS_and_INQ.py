import os
import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from keras.datasets import mnist, cifar100, cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from CompositeLayers.INQConvLayer import INQConvReluLayer
from CustomLayers.CustomLayersDictionary import customLayersDictionary
from CompositeLayers.DNSConvBNReluLayer import DNSConvBNReluLayer
from NetworkParameters import NetworkParameters
from CustomLayers.CustomLayersDictionary import customLayerCallbacks

np.random.seed(1337)  # for reproducibility


def CreateModel(input_shape, nb_classes, parameters):
    model_input = Input(shape=input_shape)

    output = model_input

    if parameters.operation_type == 'DNS':
        print('Using DNS convolution layers')
        layerType = DNSConvBNReluLayer
        kwargs = {'start_epoch' : parameters.start_epoch,
                  'gamma' : parameters.gamma,
                  'crate' : parameters.crate}

    elif parameters.operation_type == 'INQ':
        print('Using INQ convolution layers')
        layerType = INQConvReluLayer
        kwargs = {'max_value' : parameters.max_value,
                  'bit_depth' : parameters.bit_depth,
                  'start_epoch' : parameters.start_epoch,
                  'num_iterations' : parameters.num_iterations}

    else:
        assert False, 'Unsupported operation type!'

    output = layerType(input=output, nb_filters=16, border='valid', kernel_size=(2, 2), stride=(2, 2), **kwargs)
    output = layerType(input=output, nb_filters=16, border='valid', kernel_size=(2, 2), stride=(2, 2), **kwargs)
    output = layerType(input=output, nb_filters=16, border='valid', kernel_size=(2, 2), stride=(1, 1), **kwargs)
    output = layerType(input=output, nb_filters=16, border='valid', kernel_size=(2, 2), stride=(1, 1), **kwargs)
    output = layerType(input=output, nb_filters=16, border='valid', kernel_size=(2, 2), stride=(1, 1), **kwargs)

    output = Flatten()(output)
    output = Dense(nb_classes, use_bias=True, activation='softmax')(output)

    model = Model(inputs=model_input, outputs=output)

    model.summary()

    return model



############################
# Parameters

modelDirectory = os.getcwd()


parameters = NetworkParameters(modelDirectory)
parameters.nb_epochs = 15
parameters.batch_size = 32
parameters.lr = 0.0005
parameters.batch_scale_factor = 8
parameters.decay = 0.001


parameters.start_epoch = 3             # Epoch to start DNS or INQ

# DNS parameters
parameters.gamma = 0.0001
parameters.crate = 1.7                 # This value controls how aggressively to apply DNS (

# INQ parameters
parameters.max_value = 4.0
parameters.bit_depth = 5
parameters.num_iterations = 10


parameters.operation_type = 'DNS' # Either 'INQ' or 'DNS'

parameters.lr *= parameters.batch_scale_factor
parameters.batch_size *= parameters.batch_scale_factor

print('Learning rate is: %f' % parameters.lr)
print('Batch size is: %d' % parameters.batch_size)

optimiser = Adam(lr=parameters.lr, decay=parameters.decay)

############################
# Data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = np.squeeze(y_train)
y_test  = np.squeeze(y_test)

if len(X_train.shape) < 4:
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

input_shape = X_train.shape[1:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 256.0
X_test = X_test / 256.0

nb_classes = y_train.max() + 1

y_test_cat = np_utils.to_categorical(y_test, nb_classes + 1)
y_train_cat = np_utils.to_categorical(y_train, nb_classes + 1)


############################
# Training

model = CreateModel(input_shape=input_shape, nb_classes=nb_classes+1, parameters=parameters)

model.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=['accuracy'])

checkpointCallback = ModelCheckpoint(filepath=parameters.modelSaveName, verbose=1)
bestCheckpointCallback = ModelCheckpoint(filepath=parameters.bestModelSaveName, verbose=1, save_best_only=True)

model.fit(x=X_train,
          y=y_train_cat,
          batch_size=parameters.batch_size,
          epochs=parameters.nb_epochs,
          callbacks=[checkpointCallback, bestCheckpointCallback] + customLayerCallbacks,
          validation_data=(X_test, y_test_cat),
          shuffle=True,
          verbose=1
          )


print('Testing')
modelTest = load_model(filepath=parameters.bestModelSaveName, custom_objects=customLayersDictionary)

validationAccuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print('\nBest Keras validation accuracy is : %f \n' % (100.0 * validationAccuracy[1]))