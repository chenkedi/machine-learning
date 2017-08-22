# coding=utf-8

import numpy as np
import pandas as pd
import seaborn as sns

from base.algorithm.regression.linear import func


if __name__ == "__main__":

    print("读取示例数据....")

    data1 = pd.read_csv('resources/data1.txt', header=None, names=['x', 'y'])

    print("绘制一维数据的散点图...")
    sns.jointplot(x='x', y='y', data=data1)

    x = data1['x']
    y = data1['y']
    m = y.shape[0]
    # 给训练的变量加一列1，作为theta0的乘数，方便矩阵运算
    x = np.mat([np.ones(m), x])
    y = np.mat(y).transpose()
    theta = np.mat(np.zeros(x.shape[0])).transpose()

    print("初始cost值: ", str(func.compute_cost(x, y, theta)))

    print("开始进行梯度下降...")
    theta = func.gradient_decent(x, y, theta)


