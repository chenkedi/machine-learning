# coding=utf-8

import numpy as np
import pandas as pd
import seaborn as sns

from base.algorithm.regression.linear import func


if __name__ == "__main__":

    print("读取示例数据....")

    data2 = pd.read_csv('resources/data2.txt', header=None)
    print(data2.T.index)

    print("绘制一维数据的散点图...")
    # sns.jointplot(x='x', y='y', data=data2)

    x = data2.T[0:2].values
    x = func.normalize_feature(x)
    # print(x)
    # 这里为不能用data2.T[2],与dataFrame.ix(n)方式不同，直接使用方括号取值时必须指定范围
    y = data2.T[2:3].values.reshape(47,)
    # print(y)
    m = y.shape[0]  # 注意，len函数只会计算一列有多少个元素

    x = np.row_stack((np.ones(m), x))  # 也可以写作 np.r_[np.ones((1, m)), x]
    print(y.shape)
    theta = np.zeros(x.shape[0])  # 这里应该根据x的列数进行计算

    print("初始cost：", func.compute_cost(x, y, theta))

    print("开始梯度下降....")
    func.gradient_decent(x, y, theta)


