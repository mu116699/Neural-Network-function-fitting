import numpy as np

#[[1.,0.,0.],[0.,1.,2.]]numpy的维度是轴，一个维度为一个轴
#上面的为，第一轴为2，第二轴为3；

# NumPy 的数组类被调用ndarray。 它也被称为别名 array。
# 请注意，numpy.array它与标准 Python 库类不同array.array，
# 后者仅处理一维数组并提供较少的功能。

# ndarray.ndim
# 阵列的轴数（维度）。
#
# ndarray.shape
# 数组的尺寸。 这是一个整数的元组，表示每个维度中数组的大小。
# 对于具有 n 行和 m 列的矩阵，shape将是(n,m)。
# shape 因此元组的长度 是轴的数量，ndim。
#
# ndarray.size
# 数组元素的总数。 这等于元素的产物shape。
#
# ndarray.dtype
# 一个描述数组中元素类型的对象。 可以使用标准的 Python
# 类型创建或指定 dtype。 另外 NumPy 提供它自己的类型。
# numpy.int32，numpy.int16 和 numpy.float64 就是一些例子。
#
# ndarray.itemsize
# 数组中每个元素的字节大小。 例如，类型元素数组float64有
# itemsize 8（= 64/8），而其中一个类型complex32有
# itemsize 4（= 32/8）。 这相当于ndarray.dtype.itemsize。
#
# ndarray.data
# 该缓冲区包含数组的实际元素。 通常，我们不需要使用此属性，
# 因为我们将使用索引设施访问数组中的元素。
a = np.array([[ 0,  1,  2,  3,  4],
       [ 9,  1,  4,  16,  9],
       [10, 11, 12, 13, 14]])
# print(a)
# print(a.shape)
# print(a.ndim)
# print(a.itemset)
# print(a.size)
# print(type(a))
# print(a.dtype.name)
# b = np.array([6,7,8])
# print(b)
# print(type(b))
b = a[0] +np.sqrt(a[1]) +a[2]
print(a)
print(b)



