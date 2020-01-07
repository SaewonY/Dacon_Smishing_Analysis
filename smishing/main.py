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

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .model import NeuralNet
from .data_loader import make_loader, SequenceBucketCollator
from .preprocess import build_vocab, build_matrix
from .train import train_model
from .utils import ON_KAGGLE, seed_everything
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
        FASTTEXT_PATH = '../input/fasttext_bigram_vocab.pkl'
        GLOVE_PATH = '../input/glove.200D.200E.pkl'
    else:
        DATASET_PATH = '../input/dacon-smishing-analysis/'
        FASTTEXT_PATH = '../input/dacon-smishing-analysis/fasttext_bigram_vocab.pkl'
        GLOVE_PATH = '../input/dacon-smishing-analysis/glove.200D.200E.pkl'

    if args.embedding == 'fasttext':
        EMBEDDING_PATH = FASTTEXT_PATH
    elif args.embedding == 'glove':
        EMBEDDING_PATH = GLOVE_PATH
    elif args.embedding == 'mix':
        EMBEDDING_PATH = [FASTTEXT_PATH, GLOVE_PATH] 
    else:
        raise NotImplementedError

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    SEED = args.seed
    seed_everything(SEED)

    # mode to run
    if args.mode == 'train':
        train(args, output_dir, DATASET_PATH, EMBEDDING_PATH)
    elif args.mode == 'inference':
        inference(args, output_dir, DATASET_PATH, EMBEDDING_PATH)
    else:
        raise NotImplementedError


def train(args, output_dir, DATASET_PATH, EMBEDDING_PATH):    

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

    # vocab = build_vocab(list(x_train.apply(lambda x: x.split())))
    max_features = args.max_features
    print("max_features", max_features)

    tokenizer = text.Tokenizer(num_words = max_features, filters='')
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    if args.embedding != 'mix':
        with open(EMBEDDING_PATH, 'rb') as f:    
            embedding_vocab = pickle.load(f)
        embedding_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, embedding_vocab, max_features, args.vector_size)  
    else:
        with open(EMBEDDING_PATH[0], 'rb') as f:    
            fasttext_vocab = pickle.load(f)
        with open(EMBEDDING_PATH[1], 'rb') as f:    
            glove_vocab = pickle.load(f)
        fasttext_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, max_features, args.vector_size)
        glove_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, glove_vocab, max_features, args.vector_size)
        embedding_matrix = np.concatenate([fasttext_matrix, glove_matrix], axis=0)

    x_train = tokenizer.texts_to_sequences(x_train)

    lengths = torch.from_numpy(np.array([len(x) for x in x_train]))

    maxlen = lengths.max()

    x_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))

    if args.criterion == 'BCE':
        criterion = nn.BCEWithLogitsLoss()

    model = NeuralNet(embedding_matrix) 
    model.to(device)


    # validate일 경우 평균 cv score만 확인하고 종료 
    if args.validate:
        num_seeds = 3
        average_cv_score = validate(args, model, criterion, x_train_padded, lengths, y_train, output_dir, device, num_seeds)
        print(average_cv_score)
        
        sys.exit()


    splits = list(KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(x_train_padded))

    for fold_num in range(n_splits):
        
        print(f"fold_{fold_num+1} start training")

        trn_idx, val_idx = splits[fold_num]
        X_train, X_valid = x_train_padded[trn_idx], x_train_padded[val_idx]
        train_length, valid_length = lengths[trn_idx], lengths[val_idx]
        Y_train, Y_valid = y_train[trn_idx], y_train[val_idx]

        train_loader = make_loader(X_train, train_length, Y_train, maxlen=maxlen, batch_size=args.batch_size)
        valid_loader = make_loader(X_valid, valid_length, Y_valid, maxlen=maxlen, batch_size=args.batch_size, is_train=False) 

        train_model(args,
                    fold_num=fold_num,
                    model=model, 
                    train_loader=train_loader,
                    valid_loader=valid_loader, 
                    criterion=criterion, 
                    output_dir=output_dir,
                    device=device,
                    n_epochs=args.n_epochs,
                    lr=args.lr)
        print()

    return None



def validate(args, model, criterion, x_train_padded, lengths, y_train, output_dir, device, num_seeds=3):

    seeds = [(seed+1)*42 for seed in range(num_seeds)]
    splits = list(KFold(args.n_folds, shuffle=True, random_state=args.seed).split(x_train_padded))
    
    average_cv_score = 0

    for seed in seeds:
        print("seed {} training starts".format(seed))
        seed_everything(seed)
        fold_num = 0
        trn_index, val_index = splits[fold_num]

        X_train, X_valid = x_train_padded[trn_index], x_train_padded[val_index]
        train_length, valid_length = lengths[trn_index], lengths[val_index]
        Y_train, Y_valid = y_train[trn_index], y_train[val_index]

        maxlen = lengths.max()
        train_loader = make_loader(X_train, train_length, Y_train, maxlen=maxlen, batch_size=args.batch_size)
        valid_loader = make_loader(X_valid, valid_length, Y_valid, maxlen=maxlen, batch_size=args.batch_size, is_train=False) 
            
        cv_score = train_model(args,
                        fold_num=fold_num,
                        model=model, 
                        train_loader=train_loader,
                        valid_loader=valid_loader, 
                        criterion=criterion, 
                        output_dir=output_dir,
                        device=device,
                        n_epochs=args.n_epochs,
                        lr=args.lr)
        average_cv_score += cv_score / num_seeds
        print()

    return average_cv_score


def inference(args, output_dir, DATASET_PATH, EMBEDDING_PATH):

    if cuda.is_available():
        device = torch.device("cuda:0")
        print("cuda enabled")
    else:
        device = torch.device("cpu")
        print("cuda disabled")

    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))
    test_df_id = test_df['id']
    
    x_train, y_train, x_test = load_input(DATASET_PATH)
    
    # vocab = build_vocab(list(x_train.apply(lambda x: x.split())))
    max_features = args.max_features
    print("max_features", max_features)
    
    tokenizer = text.Tokenizer(num_words = max_features, filters='', lower=False)
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    if args.embedding != 'mix':
        with open(EMBEDDING_PATH, 'rb') as f:    
            embedding_vocab = pickle.load(f)
        embedding_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, embedding_vocab, max_features, args.vector_size)  
    else:
        with open(EMBEDDING_PATH[0], 'rb') as f:    
            fasttext_vocab = pickle.load(f)
        with open(EMBEDDING_PATH[1], 'rb') as f:    
            glove_vocab = pickle.load(f)
        fasttext_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, fasttext_vocab, max_features, args.vector_size)
        glove_matrix, unknown_words_fasttext = build_matrix(tokenizer.word_index, glove_vocab, max_features, args.vector_size)
        embedding_matrix = np.concatenate([fasttext_matrix, glove_matrix], axis=0)

    x_test = tokenizer.texts_to_sequences(x_test)
    test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))
    maxlen = test_lengths.max()
    x_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=maxlen))

    test_dataset = TensorDataset(x_test_padded, test_lengths)
    test_collator = SequenceBucketCollator(lambda length: test_lengths.max(), maxlen=maxlen, sequence_index=0, length_index=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collator)

    del x_test, x_test_padded, unknown_words_fasttext, tokenizer
    gc.collect()
    
    # model_lists = [file for file in os.listdir(output_dir) if file.endswith('.pt')]
    fold1_model_lists = sorted([file for file in os.listdir(output_dir) if file.startswith('fold_1')])
    fold2_model_lists = sorted([file for file in os.listdir(output_dir) if file.startswith('fold_2')])
    fold3_model_lists = sorted([file for file in os.listdir(output_dir) if file.startswith('fold_3')])
    fold4_model_lists = sorted([file for file in os.listdir(output_dir) if file.startswith('fold_4')])
    fold5_model_lists = sorted([file for file in os.listdir(output_dir) if file.startswith('fold_5')])
    total_model_list = [fold1_model_lists, fold2_model_lists, fold3_model_lists, fold4_model_lists, fold5_model_lists]

    checkpoint_weights = np.array([2 ** epoch for epoch in range(args.n_epochs)])
    checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()
    
    model = NeuralNet(embedding_matrix) 
    model.to(device)
    model.eval()

    total_preds = []

    for fold_model_lists in total_model_list:

        fold_preds = []

        for file_name in fold_model_lists:

            model_path = os.path.join(output_dir, file_name)
            model.load_state_dict(torch.load(model_path))

            test_preds = np.zeros((len(test_dataset), 1))

            with torch.no_grad():
                for i, inputs in enumerate(test_loader):
                    inputs[0] = inputs[0].to(device)
                    outputs = model(inputs[0])
                    test_preds[i * args.batch_size:(i+1) * args.batch_size] = sigmoid(outputs.cpu().numpy())

            fold_preds.append(test_preds)
        
        fold_preds = np.average(fold_preds, weights=checkpoint_weights, axis=0)
        total_preds.append(fold_preds)

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
    arg('--validate', action='store_true')
    arg('--output_dir', type=str, default='./output', help='model weights saved dir')
    arg('--output_path', type=str, default='./output/submission.csv', help='submission saved path') 
    arg('--seed', type=int, default=42)
    arg('--n_epochs', type=int, default=6)
    arg('--lr', type=float, default=0.005)
    arg('--batch_size', type=int, default=512)
    arg('--vector_size', type=int, default=200)
    arg('--max_len', type=int, default=236)
    arg('--max_features', type=int, default=150000)
    arg('--n_folds', type=int, default=5)
    arg('--embedding', type=str, default='fasttext')
    arg('--criterion', type=str, default='BCE')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
