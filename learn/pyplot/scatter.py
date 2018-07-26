import numpy as np
import matplotlib.pyplot as plt

# 调整散点颜色
N = 10
x = np.random.rand(N)
y = np.random.rand(N)
x2 = np.random.rand(N)
y2 = np.random.rand(N)
area = np.random.rand(N) * 1000
fig = plt.figure()
ax = plt.subplot()
ax.scatter(x, y, s=area, alpha=0.5)
ax.scatter(x2, y2, s=area, c='green', alpha=0.6)  # 改变颜色
plt.show()

#调整散点形状
N = 10
x = np.random.rand(N)
y = np.random.rand(N)
x2 = np.random.rand(N)
y2 = np.random.rand(N)
x3 = np.random.rand(N)
y3 = np.random.rand(N)
area = np.random.rand(N) * 1000
fig = plt.figure()
ax = plt.subplot()
ax.scatter(x, y, s=area, alpha=0.5)
ax.scatter(x2, y2, s=area, c='green', alpha=0.6)
ax.scatter(x3, y3, s=area, c=area, marker='v', cmap='Reds', alpha=0.7)  # 更换标记样式，另一种颜色的样式
plt.show()

#调整散点边缘
N = 10
x = [1]
y = [1]
x2 = [1.1]
y2 = [1.1]
x3 = [0.9]
y3 = [0.9]
area = [20000]
fig = plt.figure()
ax = plt.subplot()
ax.scatter(x, y, s=area, alpha=0.5, edgecolors='face')
ax.scatter(x2, y2, s=area, linewidths=[3])
ax.scatter(x3, y3, s=area, alpha=0.5, linewidths=[3], edgecolors='r')
plt.show()