import sys
sys.path.append("..")

from common.optimizers import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from better_rnnlm import BetterRnnlm

# 1.设定超参数
wordvec_size = 650
hidden_size = 650 # RNN隐藏状态向量维度
batch_size = 20
time_size = 35 # Truncated BPTT 块大小
max_epoch = 40
lr = 20.0
max_grad = 0.25
dropout = 0.5

# 2.读入数据-预处理(准备训练样本 -  sample(raw + label) one-hot)
# 这里原始拿到的都是id
# corpus其实是corpus_train corpus_test就是测试集
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
corpus_val, _, _ = ptb.load_data('val')

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
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 4.训练(应用梯度剪裁进行学习)
best_ppl = float('inf')
for epoch in range(max_epoch):
    # 当前epoch训练
    trainer.fit(xs, ts, 1, batch_size, time_size, max_grad, eval_interval=20)

    # 忘记training data的记忆
    # 因为下面要重新infer 新的序列 新的记忆
    # 参数学到了就行 模型的记忆得忘了
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('valid perplexity: ', ppl)

    if ppl < best_ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4
        optimizer.lr = lr

    # 忘记val data的记忆
    # 下次重新用training data训练
    model.reset_state()
    print('-' * 50)

trainer.plot(ylim=(0, 500))

# 5.基于测试集数据进行评测(以往没有)
# 测试集数据 trainer没有见过 可以验证泛化能力
# 重置模型状态 忘记训练集的序列记忆
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)
