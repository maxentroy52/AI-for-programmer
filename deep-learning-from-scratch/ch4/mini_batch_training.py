import sys,os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label = True)

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

train_loss_list = []

# Hyper-parameters
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(784, 100, 10)

for i in range(iter_num):
    # Obtain a mini-batch
    # 这里 每次拿到的是100个下标 就是100个sample
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask)

    # 不得不说 非常简练抽象的表达
    # 本次的训练样本我拿到 下面就开始训练
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 训练 - 美其名曰 本质就是计算loss function对于
    # 当前参数的梯度
    # Calculate a gradient
    # 它这个逻辑是这样 loss = f(x, t, theta)
    # x, t作为常数带入后
    # loss变成 loss = f(theta; x,t)
    print(x_batch.shape)
    print(t_batch.shape)
    grad = network.numerical_gradient(x_batch, t_batch)

    # Update the parameter.
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

plt.plot(range(len(train_loss_list)), train_loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()