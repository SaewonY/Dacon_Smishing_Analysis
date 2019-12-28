import re
import os 
import gc
import time
import math
import random
import pickle
import operator
import numpy as numpy
import pandas as pd
import numpy as np

from keras.preprocessing import text, sequence
import torch
from torch import nn, cuda
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau

import gensim
from gensim.models.wrappers import FastText

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score

from data_loader import SequenceBucketCollator
from preprocess import build_vocab, build_matrix, jamo_sentence
from utils import seed_everything


import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    #################################### load data ####################################
    DATASET_PATH = './input/'
    fasttext_PATH = './input/fasttext_200.bin'
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))

    SEED = 42
    seed_everything(SEED)

    use_cuda = cuda.is_available()
    if use_cuda:
        print("cuda enabled")
    else:
        print("cuda disable")

    #################################### preprocess text ####################################
    model = FastText.load_fasttext_format(fasttext_PATH)
    fasttext = model.wv
    del model

    fasttext_vocab = {}

    for word in fasttext.vocab.keys():
        fasttext_vocab[word] = fasttext[word]

    fasttext_word_list = fasttext_vocab.keys()
    fasttext_word_vectors_list = fasttext_vocab.values()


    vocab = build_vocab(list(train_df['jamo'].apply(lambda x: x.split())))

    # train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))
    # test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))


    # print(train_df.loc[0, 'jamo'])
    # print(train_df.head())


    x_train = train_df['jamo']
    x_test = test_df['jamo']
    y_train = torch.tensor(train_df['smishing']).float().unsqueeze(1)

    max_features = len(vocab)

    tokenizer = text.Tokenizer(num_words = max_features, filters='', lower=False)
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    fasttext_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, max_features)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    lengths = torch.from_numpy(np.array([len(x) for x in x_train]))
    test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))

    maxlen = lengths.max()

    X_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))
    X_test_padded = torch.from_numpy(sequence.pad_sequences(X_test, maxlen=maxlen))

    del x_train, x_test
    gc.collect()


    #################################### training ####################################
    n_splits = 5
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X_train_padded))

    for fold_num in range(1, n_splits+1):
        print("fold_{} start training".format(fold_num))

        trn_idx, val_idx = splits[fold_num]

        X_train, X_valid = X_train_padded[trn_idx], X_train_padded[val_idx]
        train_lengths, valid_lengths = lengths[trn_idx], lengths[val_idx]
        Y_train, Y_valid = y_train[trn_idx], y_train[val_idx]

        train_dataset = TensorDataset(X_train, train_lengths, torch.tensor(Y_train))
        valid_datatset = TensorDataset(X_valid, valid_lengths, torch.tensor(Y_valid))

        train_collator = SequenceBucketCollator(lambda length: train_lengths.max(), 
                                                    sequence_index=0, 
                                                    length_index=1, 
                                                    label_index=2)

        batch_size = 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)

