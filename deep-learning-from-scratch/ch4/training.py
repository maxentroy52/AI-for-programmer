import sys,os
sys.path.append(os.pardir)
import numpy as np
from common import*

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # init with a gaussian distribution

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

def test():
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    label = np.argmax(p)
    print(label)

    t = np.zeros(p.shape, dtype=int)
    t[label] = 1
    print(t)  # [0 1 0]

    loss = net.loss(x, t)
    print(loss)

    # net.loss总共有3个参数x, t, w.
    # 对于每一个样本，(x,t)都是确定的。
    # 此时，当前的w可以用x算出y，这是推理的过程。
    # training就是，把x,t看做常数。w是变量，此时的w是一个position
    # 然后算出当前的一个梯度grad(w)
    #
    # 这个损失函数你也好理解
    # 它是可以带入一个，也可以带入所有
    # 带入所有的(x, t)之后，形成关于w的函数
    f = lambda w: net.loss(x, t)

    # 这个实现整体看也比较特殊
    # 样本 1个
    # 参数 确定
    # 参数，在当前样本，可以梯度下降一次
    # 这个例子没有梯度下降，只是算了下微分
    dW = numerical_gradient_multi_array(f, net.W)
    print(dW)

test()