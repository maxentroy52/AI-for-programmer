import sys
import numpy as np
sys.path.append("..")
import matplotlib.pyplot as plt

from common.optimizers import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 1.设定超参数
word_vec_size = 100
hidden_size = 100

max_epoch = 100
batch_size = 10
time_size = 5 # Truncated BPTT 块大小
lr = 0.1
