import os
import re
import json

from tqdm import tqdm
import numpy as np

import utils

LANS = ['deen-de', 'deen-en']

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Data', cur_path)

test_file_path = f'{data_path}/test-full'
train_file_path = f'{data_path}/training-full'


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
        raise ValueError('mistake.')
    pads = [pad_id for _ in range(pad_len)]
    return id_list + pads


def create_vocabulary(lan_filenames, threshold):
    with open(f'{lan_filenames}.txt', 'r') as first_lan:
        vocab = None
        acc_words = []
        for idx, elem in enumerate(tqdm(first_lan)):
            listed_words = list(filter(None, sentence2wordlist(elem)))
            if len(listed_words) >= 48:
                continue
            acc_words.extend(listed_words)
            if idx % 4400000 == 0:
                vocab = utils.create_vocabulary(acc_words, vocab, sorted_vocab=False)
                acc_words = []
        if len(acc_words) > 0:
            vocab = utils.create_vocabulary(acc_words, vocab, sorted_vocab=False)
            del acc_words

    all_words = list(vocab.keys())
    test = strip_alfanum(all_words)
    print(test)
    del all_words

    vocab = utils.create_vocabulary([], vocab)
    print(len(vocab))
    print(vocab[-5:])
    vocab = utils.remove_infrequents(vocab, threshold=20)
    print(len(vocab))
    vocab = utils.remove_infrequents(vocab, threshold=30)
    print(len(vocab))
    vocab = utils.remove_infrequents(vocab, threshold=40)
    print(len(vocab))
    vocab = utils.remove_infrequents(vocab, threshold=50)
    print(len(vocab))
    raise ValueError('nesto nesto')
    vocab = utils.list2dict(vocab)

    with open(f'{lan_filenames}_vocab.json', 'w') as fp:
        json.dump(vocab, fp)
    return vocab


def id_sentences_and_pad(lan_filenames, vocab):
    with open(f'{lan_filenames}.txt', 'r') as first_lan:
        all_sentences_lenghts = []
        all_sentences = []
        for elem in tqdm(first_lan):
            listed_words = list(filter(None, sentence2wordlist(elem)))

            if len(listed_words) > 48:
                continue
            all_sentences_lenghts.append(len(listed_words) + 2) # to account for start and stop tokens

            listed_words = utils.wordlist2ids(listed_words, vocab)
            listed_words = pad_sentence(add_start_end_token(listed_words, vocab), vocab, 50)

            all_sentences.append(listed_words)
    del vocab

    for idx, elem in enumerate(all_sentences):
        all_sentences[idx] = np.array(elem).reshape((1, 50))

    sents = np.vstack(all_sentences)
    del all_sentences

    all_sentences_lenghts = np.array(all_sentences_lenghts)

    print(sents.shape)
    np.savez(f'{lan_filenames}_sents.npz', sentences=sents, lengths=all_sentences_lenghts)


if __name__ == '__main__':
    lan_filenames = f'{train_file_path}/{LANS[0]}'

    use_training_vocab = False

    if use_training_vocab:
        print('Loading vocabulary.')
        train_filenames = f'{train_file_path}/{LANS[0]}'
        with open(f'{train_filenames}_vocab.json', 'r') as fp:
            vocab = json.load(fp)
    else:
        print('Creating vocabulary.')
        vocab = create_vocabulary(lan_filenames, threshold=50)

    print('Id-ing sentences and padding.')
    id_sentences_and_pad(lan_filenames, vocab)
