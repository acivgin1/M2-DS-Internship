import operator

import numpy as np


def create_vocabulary(book):
    vocabulary = {}
    for word in book:
        vocabulary[word] = vocabulary[word] + 1 if word in vocabulary else 1
    return sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)


def remove_infrequents(vocabulary, threshold=5):
    threshold_idx = [idx for (idx, val) in enumerate(vocabulary) if val[1] <= threshold][0]
    for i in range(len(vocabulary) - 1, threshold_idx - 1, -1):
        del vocabulary[i]
    return vocabulary


def list2dict(vocabulary):
    vocabulary = dict(vocabulary)
    for idx, val in enumerate(vocabulary):
        vocabulary[val] = idx
    return vocabulary


def word2onehot(word, vocabulary, eos=False):
    idx = vocabulary[word] if word in vocabulary else len(vocabulary)
    one_hot = np.zeros((len(vocabulary) + 2, 1))  # 2 is for UNK and EOS tokens
    idx = idx if not eos else one_hot.shape[0] - 1
    one_hot[idx] = 1
    return one_hot


def embedding2word(embedding_vec, embedding_matrix, vocabulary):
    # not sure if need to norm the vectors, but just in case here is the code
    embedding_vec = embedding_vec / np.linalg.norm(embedding_vec)
    embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
    cosines = np.dot(embedding_vec, embedding_matrix.transpose())
    idx = np.argmax(cosines)
    return vocabulary.items()[idx]