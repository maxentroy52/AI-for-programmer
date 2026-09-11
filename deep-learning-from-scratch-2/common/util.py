# coding: utf-8

import numpy as np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def preprocess(text):
    # 这一块本质是tokenizer的过程
    # words本质是token id list.
    # vocab means unique token id list.
    # corpus并不是token id list 因为它并不唯一
    # 它是原始文本的token id表达
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # In NLP, a corpus (plural: corpora) is a large collection of texts.
    # So, we return all the id list.
    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word

def create_to_co_matrix(corpus, vocab_size, window_size=1):
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    corpus_size = len(corpus)

    # 统计的本质 就是计数
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            # 这里要小心下标 容易弄错
            # word_id 这个东西 存在corpus里面
            # 它在corpus里面不是下标 他是元素
            # 但在vocab当中 他是下标
            # vocab是全集 corpus只是部分语料
            # 所以co_matrix的下标可以用word_id
            # 因为后者在co_matrix中是下标
            #
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)

    return np.dot(nx, ny)

# 本质还是对词频的计算方式做优化
def ppmi(C, verbose=False, eps=1e-8):
    # 用pmi修正后的共现词频矩阵
    M = np.zeros_like(C, dtype=np.float32)

    N = np.sum(C) # 所有词频
    S = np.sum(C, axis=0) # 某个单词的词频

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

    return M

# 拿到contexts and targets的token id 表达
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0 : continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    # The neural network is fundamentally doing linear algebra,
    # and linear algebra wants arrays, not lists.
    return np.array(contexts), np.array(target)

# corpus in convert_one_hot is a parameter name, not "the corpus."
# It just means "some array of word IDs."
# 1D case: target
# target = [1, 2, 3, 4, 1, 5]
# 2D case: contexts
# From the same function:
#
# python
# contexts = [[0, 2],      # sample 0: context of "say" is ["you", "goodbye"]
#             [1, 3],      # sample 1: context of "goodbye" is ["say", "and"]
#             [2, 4],      # sample 2
#             [3, 1],      # sample 3
#             [4, 5],      # sample 4
#             [1, 6]]      # sample 5
# # shape (6, 2)
# This is a 2D array — one row per sample, and each row has multiple word IDs (the context words).

def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    if corpus.ndim == 1:
        # N是样本数
        # one hot的维度就是vocab size
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def test():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    target = corpus[1:-1]
    print(corpus)
    print(target)
    print(word_to_id)
    print(id_to_word)

    # corpus理论上来说是 vocab的子集
    vocab_size = len(word_to_id)
    C = create_to_co_matrix(corpus, vocab_size)
    c0 = C[word_to_id['you']]
    c1 = C[word_to_id['i']]
    print(cos_similarity(c0, c1))

    print(C)

    W = ppmi(C)
    print('-----------------W-----------------')
    print(W)

    # SVD
    U, S, V = np.linalg.svd(W)

    print('-----------------U-----------------')
    print(U)
    #print(S)

    print('test create_contexts_target')
    contexts, target = create_contexts_target(corpus, window_size=1)
    print(contexts)
    print(target)

    print('test one hot')
    vocab_size = len(word_to_id)
    target_onehot = convert_one_hot(target, vocab_size)
    print(target_onehot)
    contexts_onehot = convert_one_hot(contexts, vocab_size)
    print(contexts_onehot)

test()