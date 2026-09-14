import sys
sys.path.append("..")

from common.util import preprocess, create_contexts_target, convert_one_hot
from simple_cbow import SimpleCBOW
from common.optimizers import Adam
from common.trainer import Trainer

# 1.设定超参
window_size = 1
hidden_size = 3
batch_size = 3
max_epoch = 1000
eval_interval = 1

# 2.读入数据-预处理(准备训练样本 -  sample(raw + label) one-hot)
text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# 3.模型构建和优化器
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4.开始训练
trainer.fit(contexts, target, max_epoch, batch_size, eval_interval)
trainer.plot()

# 5.打印模型
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])