from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers.normalization import  BatchNormalization  as bn
from keras.layers.pooling import MaxPooling1D as pool
from keras.layers.convolutional import Conv1D as cnn1
import numpy as np

# This returns a tensor
inputs = Input(shape=(72,1))
import  keras.utils

# a layer instance is callable on a tensor, and returns a tensor
x = cnn1(64,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = cnn1(64,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = pool()(x)
x = cnn1(128,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = cnn1(128,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = pool()(x)
x = cnn1(256,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)

x = cnn1(256,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)

x = cnn1(256,3)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = pool()(x)
# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)

# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)
# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)
# x = pool()(x)

# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)
# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)
# x = cnn1(512,3)(inputs)
# x = bn()(x)
# x = Activation('relu')(x)
# x = pool()(x)
x = Flatten()(x)
x = Dense(512)(x)
x = bn()(x)
x = Activation('relu')(x)
predictions = Dense(1, activation='linear')(x)

#make data
train_x = np.random.random((1000,72,1))
train_y = train_x.sum(axis=1)

# validation_x = np.random.random((100,2))
# validation_y = validation_x.sum(axis=1)

test_x = np.random.random((100,72,1))
test_y = test_x.sum(axis=1)

print(train_x.shape)
print(train_y.shape)


# This creates a model that includes
# the Input layer and three Dense layers
model = Model(input=inputs, output=predictions)

#打印模型
model.summary()

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
             metrics=['mae', 'acc'])

model.fit(train_x,train_y, validation_data=(test_x, test_y),
          nb_epoch=40, batch_size=100)

# model.save_weights('/home/etcp/szx/flower_data/third_park_predict.h5')