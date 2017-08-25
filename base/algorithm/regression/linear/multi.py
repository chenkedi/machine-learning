# coding=utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from base.algorithm.regression.linear import func


if __name__ == "__main__":

    print("读取示例数据....")

    data2 = pd.read_csv('resources/data2.txt', header=None)
    # print(data2.T)

    x = data2.T[0:2].values
    x = func.normalize_feature(x)
    # print(x)
    # 这里为不能用data2.T[2],与dataFrame.ix[n]方式不同，直接使用方括号按行索引取值，时必须指定范围，否则视为按列取
    y = data2.T.ix[2].values
    print(y)
    m = y.shape[0]  # 注意，len函数只会计算一列有多少个元素

    x = np.row_stack((np.ones(m), x))  # 也可以写作 np.r_[np.ones((1, m)), x]
    print(y.shape)
    theta = np.zeros(x.shape[0])  # 这里应该根据x的列数进行计算

    print("初始cost：", func.compute_cost(x, y, theta))

    print("开始梯度下降....")
    theta = func.gradient_decent(x, y, theta)

    print("最终拟合图3D")
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 1, 0.01)
    Y = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(X, Y)  # 将向量变为方阵

    def f(x, y):
        return theta[0] + theta[1] * x + theta[2] * y

    ax.plot_surface(X, Y, f(X, Y), rstride=1, cstride=1)
    ax.scatter(x[1, :], x[2, :], y, marker='*', color='r')
    ax.view_init(elev=30, azim=125)
    sns.plt.show()


