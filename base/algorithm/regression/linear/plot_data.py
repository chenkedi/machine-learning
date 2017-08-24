# coding=utf-8

import seaborn as sns
import matplotlib as mt
import pandas as pd
import numpy as np
import sys

if __name__ == "__main__":
    # 一维线性回归数据
    data1 = pd.read_csv('resources/data1.txt', header=None, names=['x', 'y'])
    print(data1)

    # 二维线性回归数据
    data2 = pd.read_csv('resources/data2.txt', header=None, names=['x', 'y', 'z'])
    print(data2)

    sns.jointplot(x='x', y='y', data=data1)
    sns.jointplot(x='x', y='y', data=data2)
    sns.plot([1,2,3,4])
    sns.plt.show()


def plot(data, x_name='x', y_name='y'):
    sns.jointplot(x=x_name, y=y_name, data=data)
    sns.plt.show()


