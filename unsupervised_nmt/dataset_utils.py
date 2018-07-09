import os
import re
import json

import numpy as np
import tensorflow as tf

import utils

LANS = ['deen-de', 'deen-en']

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Data', cur_path)

test_file_path = f'{data_path}/test-full/lower_case'
train_file_path = f'{data_path}/training-full'


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


text_filename = f'{test_file_path}/{LANS[0]}.txt'

with open(text_filename, 'r') as first_lan:
    vocab = None
    acc_words = []
    for idx, elem in enumerate(first_lan):
        listed_words = list(filter(None, sentence2wordlist(elem)))
        if len(listed_words) >= 48:
            continue
        acc_words.extend(listed_words)
        if idx % 10000 == 0:
            vocab = utils.create_vocabulary(acc_words, vocab)
            acc_words = []
    if len(acc_words) > 0:
        vocab = utils.create_vocabulary(acc_words, vocab)
        del acc_words

vocab = dict(vocab)
all_words = list(vocab.keys())
test = strip_alfanum(all_words)
del all_words

vocab = utils.create_vocabulary([], vocab)

print(len(vocab))
print(vocab[-5:])
vocab = utils.remove_infrequents(vocab, threshold=5)
print(len(vocab))
vocab = utils.list2dict(vocab)

with open(text_filename, 'r') as first_lan:
    all_sentences = []
    for elem in first_lan:
        listed_words = list(filter(None, sentence2wordlist(elem)))
        listed_words = utils.wordlist2ids(listed_words, vocab)
        all_sentences.append(listed_words)

for idx, sents in enumerate(all_sentences):
    if len(sents) >= 48:
        continue
    all_sentences[idx] = pad_sentence(add_start_end_token(sents, vocab), vocab, 50)

with open(f'{test_file_path}/{LANS[0]}_vocab.json', 'w') as fp:
    json.dump(vocab, fp)
    del vocab

sents = np.array(all_sentences)
del all_sentences

print(test)
print(sents.shape)
np.save(f'{test_file_path}/{LANS[0]}_sents.npy', sents)
