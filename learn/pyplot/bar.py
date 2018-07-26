import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {'topics': ['Excel', 'Hadoop', 'Matlab', 'R语言'],
        'numbers': [654, 352, 431, 678]}

df = pd.DataFrame(data).set_index('topics')
print(df)

"""
柱状图示例
"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure(figsize=(10, 6))

# 设置x轴柱子的个数
x = np.arange(4) + 1
print(x)

# 设置y轴的数值，需将numbers列的数据先转化为数列，再转化为矩阵格式
y = list(df['numbers'])
print(y)

# 构造不同课程类目的数列
xticks1 = list(df.index)
print(xticks1)

# 画出柱状图
plt.bar(x, y, width=0.3, align='center', color='b', alpha=0.8)  # tick_label=xticks1) 这里也可以简单设置X轴每个柱子的标签

# 设置x轴的刻度，将构建的xticks代入，同时由于课程类目文字较多，在一块会比较拥挤和重叠，因此设置字体和对齐方式
plt.xticks(x, xticks1, size='small', rotation=30)

# x、y轴标签与图形标题
plt.xlabel('课程主题类别')
plt.ylabel('参与人数')
plt.title('不同课程的平均学习人数')

# 设置数字标签
for a, b in zip(x, y):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)

# 设置y轴范围
plt.ylim(0, 1000)
plt.show()
