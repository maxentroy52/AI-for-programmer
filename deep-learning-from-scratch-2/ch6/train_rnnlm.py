import sys
sys.path.append("..")

from common.optimizers import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from rnnlm import Rnnlm

# 1.设定超参数
wordvec_size = 100
hidden_size = 100 # RNN隐藏状态向量维度
batch_size = 20
time_size = 35 # Truncated BPTT 块大小
max_epoch = 4
lr = 20
max_grad = 0.25

# 2.读入数据-预处理(准备训练样本 -  sample(raw + label) one-hot)
# 这里原始拿到的都是id
# corpus其实是corpus_train corpus_test就是测试集
corpus, word_to_id, id_to_word = ptb.load_data('train')
#corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
# 对比上一章的RNN
# 这里没有截断 使用全部训练数据
xs = corpus[:-1] # sample
ts = corpus[1:]  # label

# 样本构造这里注意
# 序列样本的构造标签怎么拿
# RNNLM的映射是这样 id -> wordvec -> hidden -> vocab(logits) -> vocab(softmax)
#          (10, 5, 1)-> (10, 5, 100) -> (10, 5, 100) -> (10, 5, vocab_size) -> (10, 5, vocab_size)

# 3.生成模型
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 4.训练
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))