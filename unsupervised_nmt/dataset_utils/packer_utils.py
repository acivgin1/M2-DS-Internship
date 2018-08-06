import re
import operator

import numpy as np


def fill_vocabulary(book, vocabulary={}, sorted_vocab=True):
    for word in book:
        vocabulary[word] = vocabulary[word] + 1 if word in vocabulary else 1
    return sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True) if sorted_vocab else vocabulary


def remove_infrequents(vocabulary, threshold):
    threshold_idx = [idx for (idx, val) in enumerate(vocabulary) if val[1] <= threshold][0]
    return vocabulary[:threshold_idx]


def id_dictionary(vocabulary):
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
    # reversing vocabulary
    vocabulary = {v: k for k, v in vocabulary.items()}
    for ids in idslist:
        if ids in vocabulary:
            sent.append(vocabulary[ids])
    return sent


def sentences2ids(sentences, vocabulary):
    for idx, sents in enumerate(sentences):
        sentences[idx] = wordlist2ids(sents, vocabulary)
    return sentences


def sentence2wordlist(sentence):
    return re.split('[\W_]', sentence)


def strip_alfanum(sentence):
    if isinstance(sentence, list):
        sentence = ''.join(sentence)
    return list(set(re.split('[\w\n]', sentence)))


def remove_empty_char(wordlist):
    return list(filter(None, wordlist))


def add_start_end_token(id_list, vocabulary):
    start_id = len(vocabulary) + 1
    stop_id = len(vocabulary) + 2
    return [start_id] + id_list + [stop_id]


def pad_sentence(id_list, vocabulary, sent_len, is_pad_end=True):
    if is_pad_end:
        pad_id = len(vocabulary) + 2
    else:
        pad_id = len(vocabulary) + 3

    pad_len = sent_len - len(id_list)

    if pad_len < 0:
        raise ValueError(f'Parameter id_list is too long to be padded to {sent_len} lenght.')
    pads = [pad_id for _ in range(pad_len)]
    return id_list + pads
