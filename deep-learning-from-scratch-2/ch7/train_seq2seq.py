# coding: utf-8
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt

# 训练数据换了 之前都是自己从corpus(ptb数据集)里构建sequence(一个sequence一个样本)
# 这次直接使用了sequence数据集
from dataset import sequence

from seq2seq import Seq2seq
from common.optimizers import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq

# 1.设定超参数
wordvec_size = 16
hidden_size = 128 # RNN隐藏状态向量维度
batch_size = 128
max_epoch = 25
lr = 20.0
max_grad =5.0

# 2.读入数据集
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# Reverse input? =================================================
is_reverse = True  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

vocab_size = len(char_to_id)

# 3.生成模型
model = Seq2seq(vocab_size, wordvec_size, hidden_size, )
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4.训练模型
acc_list = []
for epoch in range(max_epoch):
    # 1epoch 拟合
    trainer.fit(x_train, t_train, max_epoch=1,batch_size=batch_size)


    # 1epoch 评测
    # 拟合完成之后 相当于学了一轮 参数更新下
    # 然后评测看看怎么样
    correct_num = 0
    for i in range(len(x_test)):
        question, answer = x_test[i], t_test[i]
        question = question.reshape(1, -1)
        answer = answer.reshape(1, -1)
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, answer, id_to_char, verbose, is_reverse)

    acc = float(correct_num)/len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc*100))

# 5.可视化
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()