'''培训一个简单的小点子根据 MNIST 重组。'''

from __future__ import print_function
import keras
# 使用Sequential模型
from keras.models import Sequential
# 导入Dense，Dropout，Flatten，Conv2D，MaxPooling2D层
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# 调用后端接口
from keras import backend as K
import numpy as np

# batch大小，每处理128个样本进行一次梯度更新
batch_size = 128
# 类别数
num_classes = 28
# 迭代次数
epochs = 12

# make data
x_train = np.random.random((60000,28,28))
y_train = x_train.sum(axis=1)

x_test = np.random.random((10000,28,28))
y_test = x_test.sum(axis=1)


# input image dimensions
# 28x28 图像
img_rows, img_cols = 28, 28

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

print(x_train[0])
print(y_train[0])

#函数拟合不需要y的值
# convert class vectors to binary class matrices
# # 标签转换为独热码
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建模型
model = Sequential()
# 第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为relu
# 第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(128, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax，输出为10个分类的概率
model.add(Dense(28, activation='linear'))

# # 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
             metrics=['mae', 'acc'])

#打印模型
model.summary()

# 训练模型，载入数据，verbose=1为输出进度条记录
# validation_data为验证集
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#模型评估不需要没有用，大部分预测的结果都是不一样都是有误差的。
# # 开始评估模型效果
# # verbose=0为不输出日志信息
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
