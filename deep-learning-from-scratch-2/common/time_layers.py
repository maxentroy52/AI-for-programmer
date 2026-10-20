import numpy as np

from common.layers import Embedding
from common.functions import softmax
from common.functions import sigmoid

# 这个类的backward方法 实现的非常典型 就是按照forward的数序 复合函数求导 利用chain rule
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    # forward这里的注意项
    # 1.推理计算出h_next即可 softmax + loss 这个是training需要做的
    # 按照以前的层次理解 这里就是算分 只不过实际是保留状态 获取记忆
    # 2.cache缓存的是 training计算导数时用的变量
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params

        t = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    # backward注意事项
    # 1.backward的输入就是forward的输出变量的导数
    # 2.backward的输出就是forward的输入 + 参数
    # 当然 是否真的返回这么多 不必要
    # 首先 参数不用返回
    # 其次 不用前向传播的 不用返回
    def backward(self, dh_next):
        Wx, Wh, b = self.params

        x, h_prev, h_next = self.cache

        # dt = dh_next * (partial h_hext by partial t)
        dt = dh_next * (1 - h_next**2)

        # db = dt * ( partial t by partial b )
        db = np.sum(dt, axis=0)

        # dwh = dt * (partial t by partial wh)
        dWh = np.dot(h_prev.T, dt)

        # dh_prev = dt * (partial t by partial h_prev)
        dh_prev = np.dot(dt, Wh.T)

        # dwx = dt * (partial t by partial wx)
        dWx = np.dot(x.T, dt)

        # dx = dt * (partial t by partial x)
        dx = np.dot(dt, Wx.T)

        # deep copy
        # store them in self.grads for later use.
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

# RNN class并没有真正的实现 RNN的 loop
# 只是building block 一个层的计算
# TimeRNN 增加loop
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        # RNN wrapper
        # 需要开始组织layers
        self.layers = None

        # stateful主要用来解决第一个神经元没有输入h的情况
        # 当然 stateful还有更通用的功能
        # 即对于每一个RNN层 决定是否保留上一个时刻的隐藏状态
        self.stateful = stateful

        self.h, self.dh = None, None

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params

        # 这个NTD解释一下
        # N是样本的个数
        # T是样本的序列长度 此时样本变成了一个序列
        # D是每一个序列的维度
        # 还是可以用之前的例子来说
        # [[you, goodbye],     [[0,2]      [[1,0,0,0,0,0], [0,0,1,0,0,0]
        #  [say, and]           [1,3]       [0,1,0,0,0,0], [0,0,0,1,0,0]
        #  [goodbye, i]]        [2,4]]      [0,0,1,0,0,0], [0,0,0,1,0,0]]
        # N = 3, T = 2, D = 6
        N, T, D = xs.shape

        # Wx的shape也有说法
        # H是hidden dimension 回想cbow的实现
        D, H = Wx.shape

        # 这里说一下h和hs的区别
        # self.h	hidden state at one timestep (the "carry-over memory")	(N, H)
        # hs hidden states at all T timesteps, stored for output/backprop (N, T, H)

        hs = np.empty((N, T, H), dtype = 'f')

        # self.h 本质是一个临时变量 为什么变成了成员变量
        # 因为 RNN在处理第一个timestamp的时候 没有previous hidden state
        # (N,H)shape是因为批处理N个样本 每一个hidden state维度是H
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype = 'f')

        # 注意
        # The RNN processes one timestep at a time, in a loop.
        # 所以下面的循环处理结束 本质是处理完成一个样本
        # 但是 可以批处理 理解上当成一个样本
        self.layers = []
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:], self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs

    # backward的过程刚好和forward反过来
    # forward的输入/输出 变成backward的输出/输入
    # 注意 backward这里返反过来的输出/输入 必须是导数
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        # 这个是输出的格式
        dxs = np.empty((N, T, D), dtype = 'f')

        # 每一个块bp的时候 上游截断了
        dh = 0
        grads = [0,0,0]
        for t in range(T):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:,t,:] + dh)
            dxs[:,t,:] = dx

            # 所有RNN层本质是一层 共享参数
            # 但是TimeRNN的权重梯度 是各个RNN的权重梯度之后
            # Wrapper类其实 导数没有意义 因为RNN有导数
            # 这里就按定义计算即可 累加起来
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        # 暂时不知道self.dh有啥用
        # 为啥要当成成员变量
        self.dh = dh

        return dxs

# 准备T个embedding 层
# 处理各个时刻的数据即可
# TimeEmbedding/TimeAffine看起来可以并行
# 这里实现成串行 有一些好处吧 比如backward求导时方便累加 但不是绝对原因 也可以并行
# TimeClass的好处是 作为一个wrapper 向下屏蔽实现细节 向上提供简单的接口
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        # 这里没有D 是因为这里是idx 是标量 还不是向量
        N, T= xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype = 'f')
        self.layers = []

        for t in range(T):
            layer = Embedding(W = self.W)

            # 这里t是timestamp
            # xs[t]取到第t个timestamp的输入 它是一个idx = xs[:,t,:]
            out[:, t, :] = layer.forward(xs[:,t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:,t,:])

            # Time layer作为wrapper layer
            # 导数都是这么定义的 各个层加起来
            grad += layer.grads[0]

        self.grads[0][...] = grad

        return None

# 如上所言
# TimeAffine/TimeEmbedding 可以并行 xs是一下拿到的
# TimeAffine采用了并行的方式实现 没有构造layers in serial manner
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.xs = None

    # 这个不赘述
    # 可以看一个数据例子
    # 多个矩阵合并后一起计算
    def forward(self, xs):
        # 注意
        # TimeEmbedding的forward拿到的是idx 要取embedding
        # TimeAffine拿到的就已经是Embedding了
        N, T, D = xs.shape

        W, b = self.params

        xs_new = xs.reshape(N*T, -1)
        out = np.dot(xs_new, W) + b
        self.xs = xs

        return out.reshape(N, T, -1)

    def backward(self, dout):
        xs = self.xs

        N, T, D = xs.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        xs_new = xs.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(xs_new.T, dout)
        dxs = np.dot(dout, W.T)
        dxs = dxs.reshape(*xs.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dxs

# 这个类的网络结构参考书中结构
# 有两点说明
# 1.对于时序数据来说
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        # sample 0: [i, love, cats]   → 3 real words
        # sample 1: [hi, PAD, PAD]    → 1 real word, 2 padding
        #
        # 这里是一个非常关键的点
        # 序列长度3对吧 不是每一个序列都能到这个长度
        # 那怎么统一处理？
        # Process each sample separately, one at a time, no batch. → slow, no GPU parallelism. ✗
        # Make T = 1 (the shortest) → then sample 0 can't fit. ✗
        # Make T = 3 (the longest) → sample 1 has 2 empty slots to fill. What goes there? → padding. ✓
        #
        # 所以 唯一的办法就是padding
        # 下面的代码主要就是识别出这些padding label
        # 用mask标记出来
        #
        if ts.ndim == 3:
            ts = ts.argmax(axis = 2)
        mask = (ts != self.ignore_label)

        xs = xs.reshape(N*T, V)
        ts= ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log( ys[np.arange(N * T), ts] )
        ls *= mask # ignore label损失设为0
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N,T,V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))

        # Token = one unit in the sequence (the model's atomic input/output element)
        return dx

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None # 用来缓存forward过程的中间结果 backward时使用

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        #x: (N, D) = (1, 300)
        #当前输入向量
        #h_prev: (N, H) = (1, 128)
        #上一步隐藏状态
        #c_prev: (N, H) = (1, 128)
        #上一步记忆单元

        #   (1,300)@(300,512) = (1,512)
        # + (1,128)@(128,512) = (1,512)
        # + (512,) 广播        = (1,512)
        # → A: (1, 512)
        # A 是 (1, 512)——一条流、四个门拼在一起。
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice
        # 遗忘门
        # f = A[:, :128]       → (1, 128)

        # 候选记忆
        # g = A[:, 128:256]    → (1, 128)

        # 输入门
        # i = A[:, 256:384]    → (1, 128)

        # 输出门
        #o = A[:, 384:512]    → (1, 128)

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # 保留多少旧记忆 + 写入多少新信息
        # 这样设计能解决梯度消失
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = None
        # 对比RNN 隐藏状态多了一个
        self.h, self.c = None, None
        self.dh = None

        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params

        # 这个NTD解释一下
        # N是样本的个数
        # T是样本的序列长度 此时样本变成了一个序列
        # D是每一个序列的维度
        # 还是可以用之前的例子来说
        # [[you, goodbye],     [[0,2]      [[1,0,0,0,0,0], [0,0,1,0,0,0]
        #  [say, and]           [1,3]       [0,1,0,0,0,0], [0,0,0,1,0,0]
        #  [goodbye, i]]        [2,4]]      [0,0,1,0,0,0], [0,0,0,1,0,0]]
        # N = 3, T = 2, D = 6
        N, T, D = xs.shape

        # LSTM 隐藏状态的维度
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)

            # self.c是内部状态
            # 只暴露self.h
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        # 输入拿到N,T,D
        # 输出的维度H从其他地方获取
        # 输出拿到N,T,H
        # 输入的维度D从其他地方获取
        N, T, H = dhs.shape

        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            # 所有RNN层本质是一层 共享参数
            # 但是TimeRNN的权重梯度 是各个RNN的权重梯度之后
            # Wrapper类其实 导数没有意义 因为RNN有导数
            # 这里就按定义计算即可

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh

        return dxs

    def set_state(self, h, c=None):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h, self.c = None, None