import re
import os 
import gc
import sys
import json
import time
import math
import random
import pickle
import shutil
import argparse
import operator
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain

import torch
from torch import nn, cuda
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing import text, sequence

from sklearn.model_selection import KFold, StratifiedKFold

from .model import NeuralNet
from .data_loader import make_loader, SequenceBucketCollator
from .preprocess import build_vocab, build_matrix
from .train import train_model
from .utils import ON_KAGGLE
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_input(DATASET_PATH):

    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))
    if args.debug:
        train_df = train_df[:2000]

    x_train = train_df['jamo']
    x_test = test_df['jamo']
    y_train = torch.tensor(train_df['smishing']).float().unsqueeze(1)
    
    del train_df, test_df
    gc.collect()

    return x_train, y_train, x_test


def main(args):

    if not ON_KAGGLE:
        DATASET_PATH = '../input/'
        FASTTEXT_PATH = '../input/fasttext_vocab.pkl'
    else:
        DATASET_PATH = '../input/dacon-smishing-analysis/'
        FASTTEXT_PATH = '../input/dacon-smishing-analysis/fasttext_vocab.pkl'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # mode to run
    if args.mode == 'train':
        train(args, output_dir, DATASET_PATH, FASTTEXT_PATH)
    elif args.mode == 'inference':
        inference(args, output_dir, DATASET_PATH, FASTTEXT_PATH)
    else:
        raise NotImplementedError

    # if args.train:
    #     train(args, output_dir, DATASET_PATH, FASTTEXT_PATH)

    # if args.inference:
    #     inference(args, output_dir, DATASET_PATH, FASTTEXT_PATH)


def train(args, output_dir, DATASET_PATH, FASTTEXT_PATH):    

    if cuda.is_available():
        device = torch.device("cuda:0")
        print("cuda enabled")
    else:
        device = torch.device("cpu")
        print("cuda disabled")


    if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    (output_dir / 'params.json').write_text(
       json.dumps(vars(args), indent=4, sort_keys=True))

    x_train, y_train, x_test = load_input(DATASET_PATH)

    with open(FASTTEXT_PATH, 'rb') as f:    
        fasttext_vocab = pickle.load(f)

    vocab = build_vocab(list(x_train.apply(lambda x: x.split())))

    # max_features = len(vocab) 
    # print("max_features", max_features)

    tokenizer = text.Tokenizer(num_words = args.max_features, filters='', lower=False)
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    vector_size = 200
    embedding_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, vector_size)
    
    x_train = tokenizer.texts_to_sequences(x_train)

    lengths = torch.from_numpy(np.array([len(x) for x in x_train]))

    maxlen = lengths.max() # 295

    x_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))
    # del x_train, unknown_words_fasttext, tokenizer, vocab
    # gc.collect()


    if args.criterion == 'BCE':
        criterion = nn.BCEWithLogitsLoss()

    model = NeuralNet(embedding_matrix) 
    model.to(device)
    
    n_splits = args.n_folds
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=args.seed).split(x_train_padded))
    # splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed).split(x_train_padded, y_train))

    for fold_num in range(n_splits):
        
        print("fold_{} start training".format(fold_num))

        trn_idx, val_idx = splits[fold_num]
        X_train, X_valid = x_train_padded[trn_idx], x_train_padded[val_idx]
        train_length, valid_length = lengths[trn_idx], lengths[val_idx]
        Y_train, Y_valid = y_train[trn_idx], y_train[val_idx]

        train_loader = make_loader(X_train, train_length, Y_train, maxlen=maxlen, batch_size=args.batch_size)
        valid_loader = make_loader(X_valid, valid_length, Y_valid, maxlen=maxlen, batch_size=args.batch_size, is_train=False) 

        model_save_path = output_dir / f'best_score_fold_{fold_num}.pt'

        train_model(model=model, 
                    train_loader=train_loader,
                    valid_loader=valid_loader, 
                    criterion=criterion, 
                    save_path=model_save_path,
                    device=device,
                    n_epochs=args.n_epochs,
                    learning_rate=args.lr)
        print()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return None



def inference(args, output_dir, DATASET_PATH, FASTTEXT_PATH):

    if cuda.is_available():
        device = torch.device("cuda:0")
        print("cuda enabled")
    else:
        device = torch.device("cpu")
        print("cuda disabled")

    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))
    test_df_id = test_df['id']
    
    x_train, y_train, x_test = load_input(DATASET_PATH)

    with open(FASTTEXT_PATH, 'rb') as f:    
        fasttext_vocab = pickle.load(f)
    
    vocab = build_vocab(list(x_train.apply(lambda x: x.split())))
    # max_features = len(vocab) 
    
    tokenizer = text.Tokenizer(num_words = args.max_features, filters='', lower=False)
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    vector_size = 200
    embedding_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, vector_size)
    
    x_test = tokenizer.texts_to_sequences(x_test)
    test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))
    maxlen = test_lengths.max()
    x_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=maxlen))

    test_dataset = TensorDataset(x_test_padded, test_lengths)
    test_collator = SequenceBucketCollator(lambda length: test_lengths.max(), maxlen=maxlen, sequence_index=0, length_index=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collator)

    del x_test, x_test_padded, unknown_words_fasttext, tokenizer, vocab
    gc.collect()
    
    model = NeuralNet(embedding_matrix) 
    model.to(device)
    model.eval()

    model_lists = [file for file in os.listdir(output_dir) if file.endswith('.pt')]
    print(model_lists)

    total_preds = []

    for i in range(args.n_folds):

        model_path = os.path.join(output_dir, model_lists[i])
        model.load_state_dict(torch.load(model_path))

        test_preds = np.zeros((len(test_dataset), 1))

        with torch.no_grad():
            for i, inputs in enumerate(test_loader):
                inputs[0] = inputs[0].to(device)
                outputs = model(inputs[0])
                test_preds[i * args.batch_size:(i+1) * args.batch_size] = sigmoid(outputs.cpu().numpy())

        total_preds.append(test_preds)

    result = np.mean(total_preds, axis=0).reshape(-1)

    submission = pd.DataFrame.from_dict({
        'id': test_df_id,
        'smishing': result
    })

    submission.to_csv(args.output_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'inference'])
    # arg('--train', action='store_true')
    # arg('--inference', action='store_true')
    arg('--output_dir', type=str, default='./output', help='model weights saved dir')
    arg('--output_path', type=str, default='./output/submission.csv', help='submission saved path') 
    arg('--seed', type=int, default=42)
    arg('--n_epochs', type=int, default=6)
    arg('--lr', type=float, default=None)
    arg('--batch_size', type=int, default=256)
    arg('--max_len', type=int, default=236)
    arg('--max_features', type=int, default=100000)
    arg('--n_folds', type=int, default=5)
    arg('--criterion', type=str, default='BCE')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
