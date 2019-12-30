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
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from model import NeuralNet
from data_loader import make_loader, TextDataset, BucketSampler, collate_fn
from preprocess import tokenize, load_embedding, build_vocab
from train import train_model
from utils import ON_KAGGLE
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
    y_train = train_df['smishing'].values

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

    if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    use_cuda = cuda.is_available()
    if use_cuda: print("cuda enabled")
    else: print("cuda disable")


    train(args, output_dir, DATASET_PATH, FASTTEXT_PATH, use_cuda)

    if args.inference:
        inference(args, output_dir, DATASET_PATH, FASTTEXT_PATH, use_cuda)
    



def train(args, output_dir, DATASET_PATH, FASTTEXT_PATH, use_cuda):    

    (output_dir / 'params.json').write_text(
       json.dumps(vars(args), indent=4, sort_keys=True))

    x_train, y_train, x_test = load_input(DATASET_PATH)
    
    vocab = build_vocab(chain(x_train, x_test), args.max_features)
    
    x_train = np.array(tokenize(x_train, vocab, args.max_len))
    x_test = np.array(tokenize(x_test, vocab, args.max_len))

    with open(FASTTEXT_PATH, 'rb') as f:    
        fasttext_vocab = pickle.load(f)

    embedding_matrix = load_embedding(fasttext_vocab, vocab['token2id'])
    

    if args.criterion == 'BCE':
        criterion = nn.BCEWithLogitsLoss()

    model = NeuralNet(embedding_matrix) 
    if use_cuda: model.cuda()

    n_splits = args.n_folds
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=args.seed).split(x_train))

    for fold_num in range(n_splits):
        
        print("fold_{} start training".format(fold_num))

        trn_idx, val_idx = splits[fold_num]
        X_train, X_valid = x_train[trn_idx], x_train[val_idx]
        Y_train, Y_valid = y_train[trn_idx], y_train[val_idx]

        train_loader = make_loader(X_train, Y_train, batch_size=args.batch_size, max_len=args.max_len)
        valid_loader = make_loader(X_valid, Y_valid, batch_size=args.batch_size, max_len=args.max_len) 


        model_save_path = output_dir / f'best_score_fold_{fold_num}.pt'

        train_model(model=model, 
                    train_loader=train_loader,
                    valid_loader=valid_loader, 
                    criterion=criterion, 
                    save_path=model_save_path,
                    n_epochs=args.n_epochs,
                    use_cuda=use_cuda)
        print()



def inference(args, output_dir, DATASET_PATH, FASTTEXT_PATH, use_cuda):

    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'preprocessed_test.csv'))
    test_df_id = test_df['id']
    
    x_train, y_train, x_test = load_input(DATASET_PATH)
    
    vocab = build_vocab(chain(x_train, x_test), args.max_features)
    
    x_test = np.array(tokenize(x_test, vocab, args.max_len))

    with open(FASTTEXT_PATH, 'rb') as f:    
        fasttext_vocab = pickle.load(f)

    embedding_matrix = load_embedding(fasttext_vocab, vocab['token2id'])

    model = NeuralNet(embedding_matrix) 
    if use_cuda: model.cuda()

    test_dataset = TextDataset(x_test, maxlen=args.max_len)
    test_sampler = BucketSampler(test_dataset, test_dataset.get_keys(),
                                batch_size=args.batch_size, shuffle_data=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                shuffle=False, num_workers=0, collate_fn=collate_fn)

    model_lists = []
    for file in os.listdir(output_dir):
        if file.endswith('.pt'):
            model_lists.append(file)

    total_preds = []

    for i in range(args.n_folds):

        model_path = os.path.join(output_dir, model_lists[i])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_preds = np.zeros((len(test_dataset), 1))

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                if use_cuda:
                    inputs = inputs.cuda()

                outputs = model(inputs)
                test_preds[i * args.batch_size:(i+1) * args.batch_size] = sigmoid(outputs.cpu().numpy())
                import pdb; pdb.set_trace()

        total_preds.append(test_preds)
    
    result = np.mean(total_preds, axis=0).reshape(-1)

    submission = pd.DataFrame.from_dict({
        'id': test_df_id,
        'prediction': result
    })

    submission.to_csv(args.output_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--inference', action='store_true')
    arg('--output_dir', type=str, default='output', help='model weights saved dir')
    arg('--output_path', type=str, default='output/submission.csv', help='submission saved path') 
    arg('--seed', type=int, default=42)
    arg('--n_epochs', type=int, default=6)
    arg('--lr', type=float, default=None)
    arg('--batch_size', type=int, default=512)
    arg('--max_len', type=int, default=220)
    arg('--max_features', type=int, default=100000)
    arg('--n_folds', type=int, default=5)
    arg('--criterion', type=str, default='BCE')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
