# coding=utf-8

import numpy as np
import pandas as pd
import seaborn as sns

from base.algorithm.regression.linear import func


if __name__ == "__main__":

    print("读取示例数据....")

    data2 = pd.read_csv('resources/data2.txt', header=None)

    print("绘制一维数据的散点图...")
    # sns.jointplot(x='x', y='y', data=data2)

    x = data2.T[0:2].values
    x = func.normalize_feature(x)
    # print(x)
    y = data2.T[2:3].values  # 这里为何不能用data2.T[2],而必须用data2.T[2:3]??
    # print(y)
    m = len(y.transpose())  # 注意，len函数只会计算一列有多少个元素

    x = np.mat(np.row_stack((np.ones(m), x)))  # 也可以写作 np.r_[np.ones(m).reshape(1,m),x]
    y = np.mat(y).transpose()
    print(y.shape)
    theta = np.mat(np.zeros(x.shape[0])).transpose()  # 这里应该根据x的列数进行计算

    print("初始cost：", func.compute_cost(x, y, theta))

    print("开始梯度下降....")
    func.gradient_decent(x, y, theta)


