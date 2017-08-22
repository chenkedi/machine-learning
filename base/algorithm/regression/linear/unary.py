# coding=utf-8

import numpy as np
import pandas as pd
import seaborn as sns


def gradient_decent(x, y, theta, alpha, iteration=1500):

    """
    x: n * m
    y: 1 * m
    theta: 1 * n
    """
    m = len(y)
    # 给训练的变量加一列1，作为theta0的乘数，方便矩阵运算
    x = np.mat([np.ones(m), x])  # (n + 1) * m
    print(x.shape)
    theta = np.mat(theta).transpose()  # n * 1
    print(theta.shape)
    y = np.mat(y).transpose()  # m * 1
    print(y.shape)

    #  初始化theta的cost
    print("初始cost值: " + str(compute_cost(x, y, theta)))

    for i in range(1, iteration+1):
        delta = x.dot(x.transpose().dot(theta) - y)
        # print(delta.shape)
        theta -= alpha * delta / m
        print('第' + str(i) + '次迭代的theta: ' + str(theta.reshape(2)) + ' cost: ' + str(compute_cost(x, y, theta)))
    return theta


def compute_cost(x, y, theta):

    """
    x: 2 * m
    y: m * 1
    theta : 2 * 1
    """
    m = len(y)
    return np.sum(np.square(x.transpose().dot(theta) - y)) / (2 * m)


if __name__ == "__main__":
    print("读取示例数据....")

    data1 = pd.read_csv('resources/data1.txt', header=None, names=['x', 'y'])
    data2 = pd.read_csv('resources/data2.txt', header=None, names=['x', 'y', 'z'])

    print("绘制一维数据的散点图...")
    sns.jointplot(x='x', y='y', data=data1)

    theta = np.zeros(2)
    print("开始进行梯度下降...")
    theta = gradient_decent(data1['x'], data1['y'], theta, 0.01)


