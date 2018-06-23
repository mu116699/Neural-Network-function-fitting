import numpy as np
from keras import backend as K

#依托与mnist的测试进行函数的拟合

# make data
x_train = np.random.random((60000,3,4))
y_train = x_train.sum(axis=1)

x_test = np.random.random((10000,3,4))
y_test = x_test.sum(axis=1)


# input image dimensions
# 28x28  -----28--input,28--output
img_rows, img_cols = 3, 4

# tf或th为后端，采取不同参数顺序
if K.image_data_format() == 'channels_first':
    # -x_train.shape[0]=6000
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # -x_train.shape:(60000, 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # x_test.shape:(10000, 1, 28, 28)
    # 单通道灰度图像,channel=1
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 数据转为float32型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# # 归一化
# x_train /= 255
# x_test /= 255

#产生的数据的类型：
# x_train shape: (60000, 28, 28, 1)
# 60000 train samples
# 10000 test samples
# y_train shape: (60000,)
# 60000 train samples
# 10000 test samples

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

#选取制定函数的参数
print(x_train[0])
print(y_train[0])
print(x_train[0][0][0])
print(x_train[0][1][1])