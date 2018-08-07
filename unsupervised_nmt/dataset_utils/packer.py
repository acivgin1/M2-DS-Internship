import os
import sys
import json

import numpy as np
from tqdm import tqdm

from packer_utils import *


def create_vocabulary(lan_filename, vocab_filename, freq_thresh=10, sent_len_thresh=(5, 30), verbose=False):
    '''
    Create vocabulary from raw text file

    :param lan_filename: filename of the raw text file
    :param vocab_filename: filename of the output vocabulary file
    :param freq_thresh: minimum word frequency threshold, defaults to 40
    :param pair of minimum and maximum sentence length thresholds, defaults to (5, 40)
    :param verbose: print out logs, defaults to False

    :return: dict vocab: a dictionary of words as keys and ids as values,
             logs: a list of strings containing info of the results of this function
    '''

    count = 0
    mean = 0        
    mean_count = 1

    logs = ['####################################',
            f'lan_filename = {lan_filename}',
            f'vocab_filename = {vocab_filename}',
            f'freq_thresh = {freq_thresh}',
            f'sent_len_thresh = {sent_len_thresh}',
            '####################################\n']

    with open(lan_filename, 'r') as lan_file:
        vocab = {}
        acc_words = []
        for idx, elem in enumerate(tqdm(lan_file)):
            listed_words = list(filter(None, sentence2wordlist(elem)))

            mean = ((mean_count - 1) * mean + len(listed_words)) / mean_count
            mean_count += 1

            if not sent_len_thresh[0] <= len(listed_words) <= sent_len_thresh[1] - 2:
                continue

            acc_words.extend(listed_words)
            if idx % 4400000 == 0:
                vocab = fill_vocabulary(acc_words, vocab, sorted_vocab=False)
                acc_words = []
            count += 1

        if len(acc_words) > 0:
            vocab = fill_vocabulary(acc_words, vocab, sorted_vocab=False)
            del acc_words

    all_words = list(vocab.keys())
    test_for_mistakes = strip_alfanum(all_words)
    logs.append(f'leftover_characters = {test_for_mistakes}')
    del all_words

    vocab = fill_vocabulary([], vocab)

    logs.append(f'full_vocab_length = {len(vocab)}')
    logs.append(f'last_five_words = {vocab[-5:]}')
    logs.append(f'mean_sent_len = {mean:.2f}')
    logs.append(f'full_sent_count = {mean_count}')
    logs.append(f'filt_sent_count = {count}')

    freqs = np.array(list(map(lambda x: x[1], vocab)))

    til_threshold = freqs[freqs > freq_thresh].sum().astype(np.float32)
    res_threshold = freqs[freqs <= freq_thresh].sum().astype(np.float32)
    all_sum = freqs.sum().astype(np.float32)

    logs.append(f'words_contained = {100 * til_threshold / all_sum:.2f}%')
    logs.append(f'words_missing = {100 * res_threshold / all_sum:.2f}%')

    vocab = remove_infrequents(vocab, threshold=freq_thresh)
    logs.append(f'filt_vocab_length = {len(vocab)}')

    if verbose:
        for log in logs:
            print(log)

    vocab = id_dictionary(vocab)

    with open(vocab_filename, 'w') as fp:
        json.dump(vocab, fp)

    return vocab, logs


def id_sentences_and_pad(lan_filename, npz_filename, vocab, sent_len_thresh=(5, 30), max_sent_count=None, verbose=False):
    '''
    Ids sentences from lan_filename inplace, adds SOS and EOS token, as well as pads to full length.
    Saves it into a npz_filename file.
    Args:
        lan_filename: filename of the raw text file
        npz_filename: filename of the output vocabulary file
        vocab: dictionary of words paired with their id
        sent_len_thresh: pair of minimum and maximum sentence length thresholds, defaults to (5, 40)
        max_sent_count: max sentence count used to ensure same number of examples for both languages, defaults to None
        verbose: print out logs, default: False
    Returns:
        logs: a list of strings containing info of the results of this function
    '''
    logs = ['\n####################################',
            f'lan_filaname = {lan_filename}',
            f'npz_filename = {npz_filename}',
            f'sent_len_thresh = {sent_len_thresh}',
            f'max_sent_count = {max_sent_count}',
            '####################################\n']
    with open(lan_filename, 'r') as lan_file:
        all_sentences_lenghts = []
        all_sentences = []
        count = 0
        for elem in tqdm(lan_file):
            listed_words = list(filter(None, sentence2wordlist(elem)))

            if max_sent_count is not None and count == max_sent_count:
                break

            if not sent_len_thresh[0] <= len(listed_words) <= sent_len_thresh[1] - 2:
                continue

            listed_words = wordlist2ids(listed_words, vocab)
            all_sentences_lenghts.append(len(listed_words) + 2) # to account for start and end token

            listed_words = pad_sentence(add_start_end_token(listed_words, vocab), vocab, sent_len_thresh[1])
            all_sentences.append(listed_words)
            count += 1

    logs.append(f'used_sent_count = {count}')
    del vocab

    for idx, elem in enumerate(all_sentences):
        all_sentences[idx] = np.array(elem).reshape((1, sent_len_thresh[1]))

    sents = np.vstack(all_sentences)
    del all_sentences

    all_sentences_lenghts = np.array(all_sentences_lenghts)

    logs.append(f'sents_shape = {sents.shape}')
    logs.append(f'sents_len_shape = {all_sentences_lenghts.shape}')
    if verbose:
        for log in logs:
            print(log)

    np.savez(npz_filename, sentences=sents, lengths=all_sentences_lenghts)
    return logs


def vocab2tsv(vocab, tsv_filename):
    with open(tsv_filename, 'w') as tsv_fp:
        for idx, key in enumerate(vocab.keys()):
            tsv_fp.write(f'{key}\t{idx}\n')

        vocab_len = len(vocab)
        tsv_fp.write(f'UNK\t{vocab_len + 0}\n')
        tsv_fp.write(f'SOS\t{vocab_len + 1}\n')
        tsv_fp.write(f'PAD\t{vocab_len + 2}\n')


if __name__ == '__main__':
    cur_path = '/home/acivgin/PycharmProjects/M2-DS-internship'
    data_path = os.path.relpath(f'{cur_path}/Data/dataset_utils', cur_path)

    if len(sys.argv) > 1:
        lan = sys.argv[1]
    else:
        lan = input("Input either 'en' or 'de': ")

    train_filename = f'{data_path}/train_txt/train_{lan}.txt'
    test_filename = f'{data_path}/test_txt/test_{lan}.txt'
    logs_filename = f'{data_path}/logs_{lan}.txt'

    vocab_filename = f'{data_path}/vocab_{lan}.json'
    tsv_filename = f'{data_path}/{lan}.tsv'

    npz_train_filename = f'{data_path}/train_sents_{lan}.npz'
    npz_test_filename = f'{data_path}/test_sents_{lan}.npz'

    # Creating vocab and tsv file for tensorboard
    if os.path.isfile(vocab_filename):
        print('Loading vocabulary...')
        with open(vocab_filename, 'r') as fp:
            vocab = json.load(fp)
    else:
        print('Creating vocabulary...')
        vocab, logs = create_vocabulary(train_filename,
                                        vocab_filename,
                                        freq_thresh=8,
                                        sent_len_thresh=(7, 28))
        with open(logs_filename, 'w') as log_fp:
            for log in logs:
                log_fp.write(f'{log}\n')

    if not os.path.isfile(tsv_filename):
        vocab2tsv(vocab, tsv_filename)

    # Packing train and test data
    if not os.path.isfile(npz_train_filename):
        print('Id-ing train sentences and padding...')
        logs = id_sentences_and_pad(train_filename,
                                    npz_train_filename,
                                    vocab,
                                    sent_len_thresh=(7, 28),
                                    max_sent_count=2842596)
        with open(logs_filename, 'a') as log_fp:
            for log in logs:
                log_fp.write(f'{log}\n')
    else:
        print(f'{npz_train_filename} already exists.')

    if not os.path.isfile(npz_test_filename):
        print('Id-ing test sentences and padding...')
        logs = id_sentences_and_pad(test_filename,
                                     npz_test_filename, vocab,
                                     sent_len_thresh=(5, 40),
                                     max_sent_count=4446)
        with open(logs_filename, 'a') as log_fp:
            for log in logs:
                log_fp.write(f'{log}\n')
    else:
        print(f'{npz_test_filename} already exists.')
