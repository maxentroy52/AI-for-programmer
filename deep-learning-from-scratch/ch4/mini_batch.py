import sys,os
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def show():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize=True, one_hot_label=True)
    print(x_train.shape)
    print(x_test.shape)

    print(t_train.shape)
    print(t_test.shape)

    train_size = x_train.shape[0]
    batch_size = 10

    print(train_size, batch_size)

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print(batch_mask)
    print(x_batch.shape, x_batch.ndim)
    print(t_batch.shape, t_batch.ndim)

show()

## 这里这么理解
## loss算的是每一个样本的平均loss，所以最后需要除以batch_size
## 前面的计算过程不变是因为python具备type flexibility，可以同时处理scalar and vector
## 否则，前面的计算还得再套一个np.sum
## eg: batch_size = 5, 5个样本，5个label [2, 7, 0, 9, 4]
## 每个label去对应的样本里面，把对应的预测概率取出来，就是结果
## y[0, 2]
## y[1, 7]
## y[2, 0]
## y[3, 9]
## y[4, 4]
## [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]] 这是一个数组啊
## np.log( [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]] + 1e-7 )
## np.log遍历处理该数组
## 下面这个函数是如何处理向量数据
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum( np.log(y[np.arange(batch_size), t] + delta) ) / batch_size