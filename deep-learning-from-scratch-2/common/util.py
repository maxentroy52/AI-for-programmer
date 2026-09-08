# coding: utf-8

import numpy as np

def preprocess(text):
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

def test():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
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
    print(W)

test()