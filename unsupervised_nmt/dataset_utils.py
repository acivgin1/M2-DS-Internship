import os
import re

import pandas as pd
import tensorflow as tf

import utils

LANS = ['deen-de', 'deen-en']

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Data', cur_path)

test_file_path = '{}/test-full/lower_case'.format(data_path)

def sentence2wordlist(sentence):
    return re.split('[;:,›‹\-"„?!.%&\'‘“´­–\[\]\n\\/()£€$ ]', sentence)

def strip_alfanum(sentence):
    if isinstance(sentence, list):
        sentence = ''.join(sentence)
    return list(set(re.split('[a-zA-Z0-9\n ]', sentence)))

def remove_empty_char(wordlist):
    return list(filter(None, wordlist))

def add_start_end_token(id_list, vocabulary):
    start_id = len(vocabulary) + 1
    stop_id = len(vocabulary) + 2
    return [start_id] + id_list + [stop_id]

def pad_sentence(id_list, vocabulary, sent_len):
    pad_id = len(vocabulary) + 2
    pad_len = sent_len - len(id_list)
    if pad_len < 0:
        raise ValueError('mistake.')
    pads = [pad_id for _ in range(pad_len)]
    return id_list + pads

with open('{}/{}.txt'.format(test_file_path, LANS[1]), 'r') as first_lan:
    all_words = []
    all_sentences = []
    for elem in first_lan:
        listed_words = remove_empty_char(sentence2wordlist(elem))
        all_sentences.append(listed_words)
        all_words.extend(listed_words)

    test = remove_empty_char(strip_alfanum(all_words))

    vocab = utils.create_vocabulary(all_words)
    print(len(vocab))
    print(vocab[-5:])
    vocab = utils.remove_infrequents(vocab, threshold=5)
    print(len(vocab))
    vocab = utils.list2dict(vocab)

    all_sentences = utils.sentences2ids(all_sentences, vocab)
    fixed_sents = []
    for sents in all_sentences:
        if len(sents) >= 48:
            continue
        fixed = pad_sentence(add_start_end_token(sents, vocab), vocab, 50)
        fixed_sents.append(fixed)
