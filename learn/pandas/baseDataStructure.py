# coding=utf-8
import numpy as np
import pandas as pd

if __name__ == "__main__":

    print("## series对象")
    s = pd.Series([12, -4, 7, 9], index=['a', 'b', 'c', 'd'])  # 当不指定index则默认使用与数组相同的数字索引，最好为元素指定有意义的索引
    print(s)
    print(s.values)  # 取series对象的值
    print(s.index)  # 取series对象的索引
    print()

    print("## 取series对象中的元素")
    print(s[0])
    print(s['b'])
    print(s[0:2])
    print(s[['b', 'c']])
    print()

    print("## 用numpy数组或其他Series对象创建新的series对象")
    arr = np.arange(1, 5)
    a = pd.Series(arr)
    print(a)
    b = pd.Series(s)
    print(b)
    arr[2] = 10  # 注意，新series对象是对原来数组和对象的引用，改变源对象也会更改新对象
    print(a)
    print()

    print("## series中的数学运算与常用数学函数")
    print(s / 2)  # 普通数学运算符可以直接使用
    print(np.log(s))  # 当使用numpy中的函数时，需要将series对象实例传入
    print()

    print("## 使用函数来判断series对象中的元素的组成成分")
    a = pd.Series([1, 0, 2, 1, 2, 3], index=['white', 'white', 'blue', 'green', 'green', 'yellow'])
    print(a.unique())  # unique返回对象中values元素去重后的结果
    print(a.value_counts())  # value_counts() 返回values中去重并统计出现次数的结果
    print(a.isin([0, 3]))  # 用于判断给定的一列元素是否包含在数据结构之中
    print(a[a.isin([0, 3])])  # 与numpy一样可以直接写入方括号中用于筛选数据
    print()

    print("## 使用函数来判断series对象中的元素是否为异常值")
    a = np.log(s)
    print(a.isnull())  # 返回一个布尔类型的series对象，其值由源series对象的元素是否为NaN决定
    print(a.notnull())
    print(a[a.isnull()])  # 同样,该函数可以用作筛选元素用途
    print(a[a.notnull()])

    print("## series对象与字典数据类型的相互转换")
    dict1 = {}
