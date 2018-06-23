from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers.normalization import  BatchNormalization  as bn
from  keras.layers.pooling import MaxPooling1D as pool
import numpy as np

# This returns a tensor
inputs = Input(shape=(72,))
import  keras.utils


# a layer instance is callable on a tensor, and returns a tensor
x = Dense(256)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = bn()(x)
x = Activation('relu')(x)
predictions = Dense(1, activation='linear')(x)

#make data
# train_x1 = np.random.random((72,1000))
# train_x = np.hsplit(train_x1,1000)
#
# train_y1 = train_x1.sum(axis=0)
# train_y = np.hsplit(train_y1,1000)
#
# # print(x.shape)
# # print("x:",x)
# # print(y.shape)
# # print("\ny:",y)
# test_x1 = np.random.random((72,100))
# test_x = np.hsplit(test_x1,100)
# #print(test_x)
# test_y1 = test_x1.sum(axis=0)
# test_y = np.hsplit(test_y1,100)
# #print(test_y)
train_x = np.random.random((100000,72))
train_y = train_x.sum(axis=1)
test_x = np.random.random((10000,72))
test_y = test_x.sum(axis=1)
print(train_x.shape)
print(train_y.shape)
# This creates a model that includes
# the Input layer and three Dense layers
model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
             metrics=['mae', 'acc'])
model.fit(train_x,train_y, validation_data=(test_x, test_y),
          nb_epoch=40, batch_size=10000)