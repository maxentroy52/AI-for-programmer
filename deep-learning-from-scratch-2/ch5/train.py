import sys
import numpy as np

sys.path.append("..")
import matplotlib.pyplot as plt

from common.optimizers import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 1.设定超参数
wordvec_size = 100
hidden_size = 100 # RNN隐藏状态向量维度
batch_size = 10
time_size = 5 # Truncated BPTT 块大小
max_epoch = 100
lr = 0.1

# 2.读入数据-预处理(准备训练样本 -  sample(raw + label) one-hot)
# 这里原始拿到的都是id
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

# 样本构造这里注意
# 序列样本的构造标签怎么拿
# RNNLM的映射是这样 id -> wordvec -> hidden -> vocab(logits) -> vocab(softmax)
#          (10, 5, 1)-> (10, 5, 100) -> (10, 5, 100) -> (10, 5, 418) -> (10, 5, 418)
xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# 3.生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 4.训练
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size, eval_interval=3)
trainer.plot()