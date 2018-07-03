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

def wordlist2ids(wordlist, vocabulary):
    return [vocabulary[x] if x in vocabulary else len(vocabulary) for x in wordlist]

def sentences2ids(sentences, vocabulary):
    id_sents = []
    for sents in sentences:
        id_sents.append(wordlist2ids(sents, vocabulary))
    return id_sents