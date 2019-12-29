import re
import os 
import gc
import sys
import time
import math
import random
import pickle
import shutil
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain

import torch
from torch import nn, cuda
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gensim.models.wrappers import FastText

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from .model import NeuralNet
from .data_loader import make_loader
from .preprocess import tokenize, load_embedding, build_vocab, build_matrix, jamo_sentence
from .train import train_model
from .utils import ON_KAGGLE, seed_everything


import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('run_root')
    arg()

    args = parser.parse_args()

    run_root = Path(args.run_root)
    if run_root.exists():
        shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
    (run_root / 'params.json').write_text(
        json.dumps(vars(args), indent=4, sort_keys=True))

    #################################### load data ####################################
    DATASET_PATH = '../input/'
    fasttext_PATH = '../input/fasttext_200.bin'
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))

    SEED = 42
    seed_everything(SEED)

    n_splits = 5
    n_epochs = 4
    max_len = 220
    batch_size = 512
    max_features = 100000
    criterion = nn.BCEWithLogitsLoss()


    use_cuda = cuda.is_available()
    if use_cuda: print("cuda enabled")
    else: print("cuda disable")

    #################################### preprocess text ####################################
    model = FastText.load_fasttext_format(fasttext_PATH)
    fasttext = model.wv
    del model

    fasttext_vocab = {}

    for word in fasttext.vocab.keys():
        fasttext_vocab[word] = fasttext[word]

    fasttext_word_list = fasttext_vocab.keys()
    fasttext_word_vectors_list = fasttext_vocab.values()

    x_train = train_df['jamo']
    x_test = test_df['jamo']
    y_train = train_df['smishing'].values

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
                                      model_path=model_path, 
                                      n_epochs=n_epochs,
                                      use_cuda=use_cuda)