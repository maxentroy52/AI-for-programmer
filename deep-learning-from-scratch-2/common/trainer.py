# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import time

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

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ppl_list = None
        self.time_idx = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size//batch_size
        offsets = [ jump * i for i in range(batch_size) ]

        # 这段逻辑其实挺核心的
        # 一个time_size是一个样本
        # 通过offsets来组装不同的样本呢
        # [x0, x1, ...,x4]
        # [x500, x501,..., x504]
        # 这是一个两重循环组batch
        # 之前的trainer只有一重 因为sample直接获取
        # 序列模型 sample需要时间点拼接
        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
                # 构造[x0]
                # 构造[x500]

            # 这个相当于片内偏移
            # 但他是整个分组序列的下标 不是一个sample的下标
            self.time_idx += 1

        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size = 35,
            max_grad = None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size//(batch_size * time_size)

        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval

        print('data_size: %d | batch_size: %d | max_iters: %d | eval_interval: %d'
              % (data_size, batch_size, max_iters, eval_interval))

        model, optimizer = self.model, self.optimizer

        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                # 获取样本
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 计算梯度
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(self.model.params, self.model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(self.model.params, self.model.grads)

                # 保存Loss
                total_loss += loss
                loss_count += 1

                # 判断是否打印
                if (eval_interval is not None) and (iters % eval_interval == 0):
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time

                    print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(ppl)
                    total_loss = 0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
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

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate