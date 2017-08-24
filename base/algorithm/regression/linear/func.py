# coding=utf-8

import numpy as np


def gradient_decent(x, y, theta, alpha=0.01, iteration=1500):

    """
    :param x: n * m, n 为线性回归参数的个数
    :param y: m * 1
    :param theta: n * 1
    :return theta: n * 1
    """
    print("x.shape = ", x.shape)
    print("theta.shape = ", theta.shape)
    print("y.shape = ", y.shape)
    m = len(y)
    for i in range(1, iteration+1):
        delta = x.dot(x.transpose().dot(theta) - y)
        # print(delta.shape)
        theta -= alpha * delta / m
        print('第', str(i), '次迭代的theta: ', theta.reshape(x.shape[0]), ' cost: ', compute_cost(x, y, theta))
    return theta


def compute_cost(x, y, theta):

    """
    :param x: n * m
    :param y: m * 1
    :param theta : n * 1
    :return double
    """
    m = len(y)
    return np.sum(np.square(x.transpose().dot(theta) - y)) / (2 * m)


def normalize_feature(x):

    """
    :param x: (n-1) * m
    :return x: (n-1) * m
    """
    mean = np.apply_along_axis(np.mean, axis=1, arr=x).reshape(2, 1)
    return np.abs((x - mean)) / mean
