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


# 读入数据集
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

print(x_train)
print(t_train)
