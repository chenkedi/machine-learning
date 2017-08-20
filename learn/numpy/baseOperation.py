# coding=utf-8
import numpy as np

if __name__ == "__main__":
    a = np.arange(16).reshape(4, 4)
    b = np.array([['a', 'b'], ['c', 'd']])
    print(a, b)
    print(a.size)  # 数组总元素个数
    print(a.ndim)  # 数组轴的个数,即包含几个维度，矩阵包含行轴与列轴，所以轴数为2
    print(a.shape)  # 数组的维度描述
    print(a.dtype, b.dtype)  # 数组的数据类型
    print(a.dtype.name, b.dtype.name)  # 数据类型的名字
    print(a.itemsize, b.itemsize)  # 数组元素长度为多少个字节

    c = np.zeros((3, 3))
    d = np.ones((3, 3))
    print(c, d)

    

