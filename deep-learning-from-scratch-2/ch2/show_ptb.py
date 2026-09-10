import sys
sys.path.append('..')
from dataset import ptb
from common.util import create_to_co_matrix, ppmi
from sklearn.utils.extmath import randomized_svd

# 这两个参数是超参
window_size = 2
wordvec_size = 100

# corpus其实是根words对应的
# word_to_id or id_to_word是去重的结果
# 所以vocab size其实是去重后的结果
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

print('counting co-occurence...')
C = create_to_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI...')
W = ppmi(C)

print('calculating SVD...')

# truncated SVD(fast!)
U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:,:wordvec_size]
print(word_vecs.shape)



