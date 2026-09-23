import sys

import numpy as np
import pickle

sys.path.append("..")

from common.util import preprocess, create_contexts_target
from cbow import CBOW
from common.optimizers import Adam
from common.trainer import Trainer
from dataset import ptb

# 1.设定超参数
# cbow这个模型 参数这里需要注意下 其实也就是网络结构的问题
# window size本质是 input layer的个数
# sample size本质是 output layer的个数 不要搞混了
# 可以看模型代码 window size是用来构建in layers
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10
eval_interval = 10

# 2.读入数据-预处理(准备训练样本 -  sample(raw + label) one-hot)
text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
print("vocab_size:", vocab_size)
print("contexts size:", len(contexts))
print("target size:", len(target))

# 3.模型构建和优化器
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4.开始训练
trainer.fit(contexts, target, max_epoch, batch_size, eval_interval)
trainer.plot()

# 5.保存模型
word_vecs = model.word_vecs

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_ptb_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
