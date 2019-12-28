import re
import os
import numpy as np
import pandas as pd
from soynlp.hangle import decompose


def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embedding_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embedding_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass
    
    print("Found embeddings for {:.2%} of vocab".format(len(a) / len(vocab)))
    print("Found embedding for {:.2%} of all text".format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def build_matrix(word_index, word2vec_vocab, max_features):
    embedding_matrix = np.zeros((max_features + 1, 200))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = word2vec_vocab[word]
            except:
                unknown_words.append(word)
                
    return embedding_matrix, unknown_words



doublespace_pattern = re.compile('\s+')


def jamo_sentence(sent):

    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        try:
            len(cjj)
        except:
            return ' '
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '' for c in cjj)
        return cjj_

    sent_ = ''.join(transform(char) for char in sent)
    sent_ = doublespace_pattern.sub(' ', sent_)
    return sent_
# 'ㅇㅓ-ㅇㅣ-ㄱㅗ- ㅋㅔㄱㅋㅔㄱ ㅇㅏ-ㅇㅣ-ㄱㅗ-ㅇㅗ-'
# def jamo_sentence(sent):
#     sent = sent.replace('XXX', 'X')
#     def transform(char):
#         if char == ' ':
#             return char
#         elif char == 'X':
#             return ' X'
#         elif char == '.':
#             return '. '
#         else:
#             return char

#     sent_ = ''.join(transform(char) for char in sent)
#     sent_ = doublespace_pattern.sub(' ', sent_)
#     return sent_


if __name__ == '__main__':

    DATASET_PATH = './input/'
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'public_test.csv'))

    train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))
    test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))

    train_df.to_csv('./input/preprocessed_train.csv', index=False, encoding='utf-8')
    test_df.to_csv('./input/preprocessed_test.csv', index=False, encoding='utf-8')