# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []

        # 正常来说
        # 一次迭代 更新一次参数 那么下次forward会计算新的loss
        # 可以看上次的迭代是否变得更好
        # 但这么打 太多了
        # 所以 间隔着打 比如20次迭代算一次平均loss
        self.eval_internal = None

    def fit(self, x, t, max_epoch=10, batch_size=32, eval_internal = 20):
        data_size = len(x)
        max_iters = data_size//batch_size
        self.eval_internal = eval_internal

        total_loss = 0
        loss_count = 0

        for epoch in range(max_epoch):
            # 打乱数据
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]

            # 开始迭代
            for iters in range(max_iters):
                # 这是切片操作 连续拿
                batch_x = x[iters * batch_size: (iters + 1) * batch_size]
                batch_t = t[iters * batch_size: (iters + 1) * batch_size]

                # 1.计算梯度 更新参数
                # 这里采用了和train_custom_loop一样的写法
                # 原数的写法做了去重 以及梯度剪裁 RNN适用 DNN不必要
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1

                # 2.定期输出
                if (iters + 1) % eval_internal == 0:
                    avg_loss = total_loss / loss_count
                    print('|epoch %d | iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
                    self.loss_list.append(avg_loss)
                    total_loss = 0
                    loss_count = 0

    def plot(self):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label='train')
        plt.xlabel('iterations (x10)')
        plt.ylabel('loss')
        plt.show()