import re
import os 
import gc
import sys
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
from torch.utils.data import DataLoader

import gensim
from gensim.models.wrappers import FastText

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from model import NeuralNet
from data_loader import make_loader
from preprocess import tokenize, load_embedding, build_vocab, build_matrix, jamo_sentence
from train import train_model
from utils import ON_KAGGLE, seed_everything


import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    print(os.listdir('../input'))
    #################################### load data ####################################
    if not ON_KAGGLE:
        DATASET_PATH = '../input/'
        fasttext_PATH = '../input/fasttext_200.bin'
        train_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_train.csv'))
        test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))
    else:
        pass

    SEED = 42
    seed_everything(SEED)

    n_splits = 5
    n_epochs = 4
    max_len = 220
    batch_size = 512
    max_features = 100000
    criterion = nn.BCEWithLogitsLoss()


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

    # vocab = build_vocab(list(train_df['jamo'].apply(lambda x: x.split())))

    # train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))
    # test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))


    # print(train_df.loc[0, 'jamo'])
    # print(train_df.head())


    x_train = train_df['jamo']
    x_test = test_df['jamo']
    y_train = train_df['smishing'].values
    # y_train = torch.tensor(train_df['smishing']).float().unsqueeze(1)

    # max_features = len(vocab)

    # tokenizer = text.Tokenizer(num_words = max_features, filters='', lower=False)
    # tokenizer.fit_on_texts(list(x_train) + list(x_test))

    # fasttext_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, max_features)

    # x_train = tokenizer.texts_to_sequences(x_train)
    # x_test = tokenizer.texts_to_sequences(x_test)

    # lengths = torch.from_numpy(np.array([len(x) for x in x_train]))
    # test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))

    # maxlen = lengths.max()

    # X_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))
    # X_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=maxlen))

    # del x_train, x_test
    # gc.collect()


    from itertools import chain
    vocab = build_vocab(chain(x_train, x_test), max_features)
    embedding_matrix = load_embedding(fasttext_vocab, vocab['token2id'])
    
    x_train = np.array(tokenize(x_train, vocab))
    x_test = np.array(tokenize(x_test, vocab))
    

    #################################### training ####################################
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train))

    for fold_num in range(1, n_splits+1):
        print("fold_{} start training".format(fold_num))

        trn_idx, val_idx = splits[fold_num]
        X_train, X_valid = x_train[trn_idx], x_train[val_idx]
        Y_train, Y_valid = y_train[trn_idx], y_train[val_idx]

        train_loader = make_loader(X_train, Y_train, batch_size=batch_size, max_len=max_len)
        valid_loader = make_loader(X_valid, Y_valid, batch_size=batch_size, max_len=max_len) 

        model = NeuralNet(embedding_matrix) 
        if use_cuda:
            model.cuda()

        preds, cv_score = train_model(model, 
                                      train_loader=train_loader,
                                      valid_loader=valid_loader, 
                                      criterion=criterion, 
                                      n_epochs=n_epochs,
                                      use_cuda=use_cuda)