# coding: utf-8
import sys
sys.path.append('..')

import numpy as np
from common.layers import Softmax

"""
N = 1        batch
T = 3        encoder 时间步
H = 4        隐藏维度

hs = [[[1, 0, 1, 0],      # h_1
       [0, 2, 0, 0],      # h_2
       [0, 0, 1, 3]]]     # h_3
                        # 形状 (1, 3, 4)

h = [[1, 0, 1, 0]]        # 形状 (1, 4)

hr = [[[1, 0, 1, 0]]]     # (1, 1, 4)

t = hs * hr(广播 对应元素相乘)

t = [[[1,0,1,0],          # h_1 * h
      [0,0,0,0],          # h_2 * h
      [0,0,1,0]]]         # h_3 * h
                        # (1, 3, 4)
                        
s = np.sum(t, axis=2)
s = [[2, 0, 1]] 

a = softmax(s)
a = [[0.665, 0.090, 0.245]] (N,H) = (1,3) 
"""

# AttentionWeight 只做一件事：算「decoder 当前状态 h 对 encoder 每一步 h_s 的相关性权重 a」
class AttentionWeight():
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        # T个时刻
        # 每个时刻LSTM给出一个隐藏向量h
        N, T, H = hs.shape

        # 这一步只是跟样本对齐
        hr = h.reshape(N, 1, H)  # .repeat(T, axis=1)

        t = hs * hr

        # sum这里是看你按照哪个时间轴来做
        # 可以保持维度不变 还是H维，合并的是时间轴。
        # 也可以时间轴不变 合并的是维度。
        # 这里保持了时间轴不变 合并了维度
        s = np.sum(t, axis=2)

        a = self.softmax.forward(s)
        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh

"""
text
N = 1        batch
T = 3        encoder 时间步
H = 4        隐藏维度

hs = [[[1, 0, 1, 0],      # h_1
       [0, 2, 0, 0],      # h_2
       [0, 0, 1, 3]]]     # h_3
                          # (N,T,H) = (1, 3, 4) 
                          
a = [[0.665, 0.090, 0.245]]   # (N,T) = (1, 3)

ar = [[[0.665],
       [0.090],
       [0.245]]] 
ar是(N,T,1) 还不是(N,T,H) 后续会广播为(N,T,H)
但前提是需要先做成(N,T,1)

t = hs * ar

t = [[[1*0.665, 0*0.665, 1*0.665, 0*0.665],     # h_1 * 0.665
      [0*0.090, 2*0.090, 0*0.090, 0*0.090],     # h_2 * 0.090
      [0*0.245, 0*0.245, 1*0.245, 3*0.245]]]    # h_3 * 0.245

  = [[[0.665, 0,     0.665, 0    ],
      [0,     0.180, 0,     0    ],
      [0,     0,     0.245, 0.735]]]            # (1, 3, 4)
      
eg: 0.065是第一个时间步的权重 所以它理应乘以hs_1的每一个分量

c = np.sum(t, axis=1)

c = [[0.665 + 0 + 0,
      0 + 0.180 + 0,
      0.665 + 0 + 0.245,
      0 + 0 + 0.735]]

  = [[0.665, 0.180, 0.910, 0.735]]              # (1, 4)
"""

# 用 attention 权重 a 对 encoder 的 hs 加权求和
# 它就是 attention 的「取值」那一步——AttentionWeight 算权重，WeightSum 用权重把 hs 混合成 context
class WeightSum():
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        # a的shape是 (N,T)
        ar = a.reshape(N, T, 1)  # .repeat(T, axis=1)

        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da

# 让 decoder 每一步都能聚焦源序列的不同部分，而不是只依赖一个固定 C
# 核心：c 是「动态的、可聚焦的上下文」，替代了「固定、压缩的 C」。
class Attention():
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()

        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        c = self.weight_sum_layer.forward(hs, a)

        self.attention_weight = a
        return c

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

# 这里再强调下TimeAttention这个类的意义
# For training 目标序列已知
# 预测时 目标序列未知
# 训练时 输入和标签都是完整的T步 可以一次喂给TimeLSTM/TimeAttention
# 内部循环T次 一次算完loss
class TimeAttention():
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    # 这个函数主要是为了training
    # 否则输入无法给入hs_dec
    # predict解决预测的问题
    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)

        self.layers = []
        self.attention_weights = []

        # 这里forward纯属是为了backward
        # 记住cache 给backward用
        for t in range(T):
            layer = Attention()
            out[:,t,:] = layer.forward(hs_enc, hs_dec[:,t,:])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec