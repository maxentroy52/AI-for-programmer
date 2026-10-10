import sys
import numpy as np

sys.path.append("..")
import matplotlib.pyplot as plt

from common.optimizers import SGD
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
#print(xs)
#print(ts)

# 3.学习用的参数
# timesize是一个sample的大小 batch size是一个batch需要多少个sample
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 4.生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 5.训练
# jump在书里讨论过 mini-batch的思路是一个batch内尽可能看更多样化的样本
# 随机就体现在jump这里
jump = (corpus_size - 1) // batch_size
offsets = [ i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 获取mini batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(time_idx + offset) % data_size]
                batch_t[i, t] = ts[(time_idx + offset) % data_size]
            time_idx += 1

        # 计算梯度 更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 评估
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' %(epoch + 1, ppl))
    ppl_list.append(ppl)
    total_loss = 0
    loss_count = 0

x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()