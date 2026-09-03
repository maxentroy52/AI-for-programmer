# coding: utf-8
import sys
sys.path.append('..')

from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)

print('x[0]', x[0]) # x[0] [-0.  0.]
print('t[0]', t[0]) # t[0] [1 0 0]

# show the dataset
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

# 非线性数据
# 需要学习非线性的分割线