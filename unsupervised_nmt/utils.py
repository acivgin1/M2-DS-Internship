import operator

import numpy as np


def create_vocabulary(book, vocab=None, sorted_vocab=True):
    if vocab is None:
        vocabulary = {}
    else:
        vocabulary = vocab

    for word in book:
        vocabulary[word] = vocabulary[word] + 1 if word in vocabulary else 1

    if sorted_vocab:
        return sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
    else:
        return vocabulary

def remove_infrequents(vocabulary, threshold):
    threshold_idx = [idx for (idx, val) in enumerate(vocabulary) if val[1] <= threshold][0]
    return vocabulary[:threshold_idx]


def list2dict(vocabulary):
    vocabulary = dict(vocabulary)
    for idx, val in enumerate(vocabulary):
        vocabulary[val] = idx
    return vocabulary


def wordlist2ids(wordlist, vocabulary):
    for idx, word in enumerate(wordlist):
        wordlist[idx] = vocabulary[word] if word in vocabulary else len(vocabulary)
    return wordlist


def ids2wordlist(idslist, vocabulary):
    sent = []
    vocabulary = {v: k for k, v in vocabulary.items()}
    for ids in idslist:
        if ids in vocabulary:
            sent.append(vocabulary[ids])
    return sent


def sentences2ids(sentences, vocabulary):
    for idx, sents in enumerate(sentences):
        sentences[idx] = wordlist2ids(sents, vocabulary)
    return sentences
