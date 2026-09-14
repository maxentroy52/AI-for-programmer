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
        self.eval_interval = None

    def fit(self, x, t, max_epoch=10, batch_size=32, eval_interval = 20):
        data_size = len(x)
        max_iters = data_size//batch_size
        self.eval_interval = eval_interval

        total_loss = 0
        loss_count = 0
        print('data_size: %d | batch_size: %d | max_iters: %d | eval_interval: %d'
              % (data_size, batch_size, max_iters, eval_interval))
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
                self.model.params, self.model.grads = remove_duplicate(self.model.params, self.model.grads)

                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1

                # 2.定期输出
                if (iters + 1) % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    print('|epoch %d | iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
                    self.loss_list.append(avg_loss)
                    total_loss = 0
                    loss_count = 0

    def old_plot(self):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label='train')
        plt.xlabel('iterations (x10)')
        plt.ylabel('loss')
        plt.show()

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads