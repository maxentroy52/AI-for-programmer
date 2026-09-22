# Contains building blocks (layers) that can be stacked to create a network
# coding: utf-8
import numpy as np
from common.functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None # it's a cache variable.

    # Embedding layer的出现 本质是为了替换MalMul
    # 但是它和mat mul是一回事
    # 所以 forward/backward操作就是按照mat mul理解即可
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        # 这里forward和backward反过来
        # 前者取一行
        # 后者把dout赋值到这一行
        # 但是可能有重复的下标
        # 所以 累加到这一行
        # 本质矩阵乘法
        np.add.at(dW, self.idx, dout)

        # 这里其实没有操作数传进来
        # 所有不用反向传播它的导数给上游

        # 当你看到后面的EmbDot时 对这一层有更深的理解
        # Emb层的forward操作是取一行
        # backward操作是 根据这一行的导数 把这一行分量的导数给求出来
        # 这个理解非常重要 否则就无法理解EmbDot的backward在干什么
        # EmbDot的forward输入是行导数
        # EmbDot的backward输出也是行导数
        # 但是行导数 不是每一个变量的导数
        # 这个工作是交给Emb这一层去做的
        # 至此 这两个layer的forward/backward操作才能理解

        # 所以backward对照着看forward即可
        # backward把forward input的导数给求出来
        return None

class EmbeddingDot:
    def __init__(self, W):
        # 从这里就能看出来 EmbDot是一个wrapper
        # 这里的w是 w_out
        # Emb的作用是 w_in的matmul替换成矩阵取一行的操作
        # EmbDot的作用是，完成 dot 这里会涉及到取一行
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        # h是 w_in的计算结果 上下文 emb表示
        # idx用来取 target word对应的emb

        # 这里np.sum的使用有讲究
        # 因为是batch数据 如果np.dot 那是彻底的矩阵相乘
        # eg [[1,2,3], [2,3,4]] 这是两个h
        # eg [[1,3,5], [2,4,6]] 这是两个label
        # 期望的结果是这样
        # [1,2,3] dot [1,3,5] = 23
        # [1,3,5] dot [2,4,6] = 44
        # [23, 44] np.sum是算成这样
        # np.dot 那就是彻底两个矩阵相乘 显然不是这样的结果
        # 核心在于 这里的数据是batch不是矩阵
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        # 相乘的矩阵 缓存下来 backward会用
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        # 这一层的两个输入h and idx
        # h是有前驱的 所以算出来它的导数之后 需要继续backward
        # w是本层参数 所以它的导数一定要算出来 因为这是模型参数

        # 导数的计算法则不再赘述 虽然我也不理解 但规则记住就好
        # 可以参考P147的图例 那个非常清楚 按照那个理解
        # reshape这里 [5, 122, 86] 变成 [ [5], [122], [86] ] 否则不能广播
        # 还有一点是 dout的shape和out是一致的
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
