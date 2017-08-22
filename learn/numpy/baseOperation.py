# coding=utf-8
import numpy as np
import time
if __name__ == "__main__":

    print("###ndarray基本属性#####")
    a = np.arange(16).reshape(4, 4)
    b = np.array([['a', 'b'], ['c', 'd']])  # 注意这里python2默认是string8，而python3是string32
    print(a, b)
    print(a.size)  # 数组总元素个数
    print(a.ndim)  # 数组轴的个数,即包含几个维度，矩阵包含行轴与列轴，所以轴数为2
    print(a.shape)  # 数组的维度描述
    print(a.dtype, b.dtype)  # 数组的数据类型
    print(a.dtype.name, b.dtype.name)  # 数据类型的名字
    print(a.itemsize, b.itemsize)  # 数组元素长度为多少个字节


    print("# 常用矩阵构造函数")
    c = np.zeros((3, 3))
    d = np.ones((3, 3))
    e = np.arange(1, 17)
    f = np.arange(1, 17, 2)  # 第三个数字表示数组相邻两数的间隔
    g = np.arange(1, 11, 0.5)
    h = np.linspace(1, 13, 3)  # 第三个数字表示将数组分为几个部分
    i = np.random.random((3, 3))
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)
    print(i)

    print("# 常用算数与矩阵运算运算符")
    a = np.ones((3, 3))  # 这里不能用a重新赋值
    b = np.arange(9).reshape(3, 3)
    print(a + b)
    print(a * b)
    print(a * np.sin(b))
    print(a * np.sqrt(b))
    print(np.dot(a, b))
    print(a.dot(b))
    a *= 2
    print(a)
    a -= 1
    print(a)

    print("# 常用算数与矩阵运算运算符")
    a = np.arange(1, 5).reshape(2, 2)
    print(a.sum())
    print(a.min())
    print(a.max())
    print(a.mean())
    print(a.std())

    print("# 索引与切片机制")
    a = np.arange(9)
    print(a[0])
    print(a[-1])  # 负数索引表示从后往前索引
    a.shape = (3, 3)
    print(a[0, 1])

    print("# 一维数组切片")
    a = a.reshape(9)
    print(a[1:5])
    print(a[1:5:2])  # 第三个数表示每个多少个元素抽取一个
    print(a[::2])  # 从第一个元素开始抽取，一直到最后一个元素，每隔两个元素抽取一次
    print(a[:5:2])  # 从第一个元素开始抽取
    print(a[:5:])  # 元素间隔默认为1

    print("# 二维数组切片")
    a.shape = (3, 3)
    print(a[0, :])  # 选取第0行（包含所有列)
    print(a[:, 0])
    print(a[0:2, 0:2])  # 选取给定范围的连续行与列
    print(a[[0, 2], 0:2])  # 选取给定分为的不连续行与连续列

    print("# 二维数组行遍历")
    for row in a:
        print(row)

    print("# 二维数组元素遍历")
    for item in a.flat:
        print(item)

    print("# 二维数组库函数遍历")
    print(np.apply_along_axis(np.mean, axis=0, arr=a))  # 注意这里的axis为0是按列应用函数，而axis为1是按行应用，与通常的行列相反
    print(np.apply_along_axis(np.mean, axis=1, arr=a))

    def test(x): return x/2
    print(np.apply_along_axis(test, axis=0, arr=a))

    print("# 布尔数组")
    print(i < 0.5)
    print(i[i < 0.5])

    print("# 数组变换")
    a = np.random.random(12)
    print(a.reshape(3, 4))  # 注意这里没有改变原来的a数组，而是返回了一个新的数组
    a.shape = (3, 4)  # 这里改变了原数组
    print(a)
    print(a.ravel())  # 将a数组变回一维数组，与a.reshape(12)一样的作用
    print(a.transpose())  # 数组的转至

    print("# 二维数组的连接")
    a = np.ones((3, 3))
    b = np.zeros((3, 3))
    print(np.vstack((a, b)))  # 按照垂直方向堆叠数组
    print(np.hstack((a, b)))  # 按照水平方向堆叠数组

    print("# 一维数组的连接")
    a = np.arange(3)
    b = np.arange(3, 6)
    c = np.arange(6, 9)
    print(np.column_stack((a, b, c)))  # 将一维数组作为列堆叠
    print(np.row_stack((a, b, c)))  # 将一维数组最为行堆叠

    print("# 数组的切分")
    a = np.arange(16).reshape((4, 4))
    print(np.hsplit(a, 2))  # 从水平方向将数组分为等宽的两个部分,与hstack互为逆操作
    print(np.vsplit(a, 2))  # 从垂直方向将数组分为登高的两个部分,与vstack互为逆操作
    print(np.split(a, [1, 3], axis=1))  # 按列方向，使用给定的列索引对数组进行拆分
    print(np.split(a, [1, 3], axis=0))  # 按行方向，使用给定的行索引对数组进行拆分

    print("# 结构化数组")
    structured = np.array([
        (1, 'First', 0.5, 1+2j),
        (2, 'second', 1.3, 2-2j)],
        dtype=('i2, a6, f4, c8')
    )
    print(structured.dtype)  # 每个类型自动分配一个列名
    print(structured[1])  # 数字索引对应着某行元素
    print(structured['f0'])  # 自动分配的符号索引则对应着一列同类型的元素
    structured = np.array([
        (1, 'First', 0.5, 1+2j),
        (2, 'second', 1.3, 2-2j)],
        dtype=[('id', 'i2'),
               ('position', 'a6'),
               ('value', 'f4'),
               ('complex', 'c8')
               ]
    )  # 同样，可以手动指定列索引的名称 或者也可以在定义好数组后，再通过dtype指定
    structured.dtype.names = ('id', 'order', 'value', 'complex')
    print(structured['order'])

    print("# 读写数组二进制文件")
    a = np.random.random((3, 3))
    print(a)
    np.save('resources/binary_array', a)
    print(np.load('resources/binary_array.npy'))

    print("# 读写csv文件")
    data = np.genfromtxt('resources/data.csv', delimiter=',', names=True)
    print(data)

    print("# 读写csv文件, 并处理缺失值")
    data = np.genfromtxt('resources/dataWithNaN.csv', delimiter=',', names=True)
    print(data)
    np.savetxt('resources/dataWrittenByNumpy', data)

    # print("# 改变numpy默认的行向量为列向量模式，加快向量复制操作")
    # a = np.asfortranarray(np.random.rand(5000, 5000, 3))
    # tic = time.time()
    # a[:, :, 0] = a[:, :, 1]
    # a[:, :, 2] = a[:, :, 0]
    # a[:, :, 1] = a[:, :, 2]
    # toc = time.time() - tic
    # print(toc)

    a = np.array([1,2,3])
    print(np.square(a.reshape(3, 1)))


