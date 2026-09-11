import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 初始化权重参数
        W_in = np.random.randn(V, H).astype('float32')
        W_out = np.random.randn(H, V).astype('float32')

        # 生成层
        # in_layer0 and in_layer1
        # 共享参数 trainer需要考虑去重
        # self.params will contain W_in twice (once from each layer), so the optimizer must deduplicate.
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度 统一管理
        # 这是模型的核心参数
        # 统一放在这里 便于交给optimizer
        layers = [self.in_layer0, self.in_layer1, self.out_layer] # 这是个临时参数
        self.params, self.grads = [], [] # 特别注意 这是模型参数汇总 放在 python list里面 不是numeric array 不参与计算
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # mini-batch contexts是三维数组
        # 第一维是sample id
        # 第二维度window size
        # 正常来说 一个样本 contexts[0]是一个vector contexts[1]是一个vector
        # contexts[0]和in_layer0 dot product
        # contexts[1]和in_layer1 dot product
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])

        # 这里获取中间层结果 也就是encoder 之后的信息
        h = (h0 + h1) * 0.5

        # 然后走到decoder 与wout dot product
        score = self.out_layer.forward(h)

        # 这里不返回score也是有原因的
        # cbow一般不用来做推理 只用来做w_in数组 或者 word_vec的获取
        # 所以一般来说forward不关心score 只关心loss
        # 所以 这也是为什么输入里有一个target
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout = 1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da = da * 0.5

        # 下面两层是最后两层
        # 因为没有需要在向前传播的意义
        # 所以返回值不care
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)

        return None