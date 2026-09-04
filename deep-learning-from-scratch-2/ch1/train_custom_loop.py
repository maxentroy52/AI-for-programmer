# coding: utf-8
import sys

from setuptools.windows_support import hide_file

sys.path.append('..')
import numpy as np
from dataset import spiral
import matplotlib.pyplot as plt

from common.optimizers import SGD
from two_layer_net import TwoLayerNet

# 1. 设定超参数
max_epoch = 300
batch_size = 30
learning_rate = 1.0

# 2.读入数据
x, t = spiral.load_data()

# 3.构建模型和优化器
input_size = 2
hidden_size = 10
output_size = 3
model = TwoLayerNet(input_size, hidden_size, output_size)
optimizer = SGD(learning_rate)

# 4.学习用的变量
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

# 5. 开始训练
for epoch in range(max_epoch):
    # 打乱数据
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]


    for iters in range(max_iters):
        # 这是切片操作 连续拿
        batch_x = x[iters * batch_size : (iters + 1) * batch_size]
        batch_t = t[iters * batch_size : (iters + 1) * batch_size]

        # 1.计算梯度 更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

        # 2.定期输出
        if (iters + 1) % 5 == 0:
            avg_loss = total_loss / loss_count
            print('|epoch %d | iter %d / %d | loss %.2f'%(epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss = 0
            loss_count = 0


## --------------------------- drawing code -----------------------------------

# 学習結果のプロット
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()