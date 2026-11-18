# coding: utf-8
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence

from common.optimizers import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq

from attention_seq2seq import AttentionSeq2seq

# 1.读入数据
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 反转输入语句
x_train, x_test = x_train[:,::-1], x_test[:,::-1]

# 2.设定超参数
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256 # RNN隐藏状态向量维度
batch_size = 128
max_epoch = 10
max_grad =5.0

# 3.生成模型
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4.训练模型
acc_list = []
for epoch in range(max_epoch):
    # 1epoch训练
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size)

    # 训练完成 更新一轮参数 评估结果
    correct_num = 0
    for i in range(len(x_test)):
        question, answer = x_test[i], t_test[i]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, answer, id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# 5.模型下载
model.save_params()

x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()
