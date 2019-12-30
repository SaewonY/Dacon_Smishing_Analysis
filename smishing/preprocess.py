import re
import os
import numpy as np
import pandas as pd
from collections import Counter
# from .utils import ON_KAGGLE 
# if not ON_KAGGLE:
#     from soynlp.hangle import decompose

max_features = 100000


def tokenize(texts, vocab, max_len):
    
    def text2ids(text, token2id):
        return [
            token2id.get(token, len(token2id) - 1)
            for token in text.split()[:max_len]]
    
    return [
        text2ids(text, vocab['token2id'])
        for text in texts]


def load_embedding(embedding, word_index):

    embed_size = 200

    # word_index = tokenizer.word_index
    nb_words = min(max_features + 2, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))


    for key, i in word_index.items():
        word = key
        embedding_vector = embedding.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_vocab(texts, max_features):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {
        'token2id': {'<PAD>': 0, '<UNK>': max_features + 1},
        'id2token': {}
    }
    vocab['token2id'].update(
        {token: _id + 1 for _id, (token, count) in
         enumerate(counter.most_common(max_features))})
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    return vocab



if __name__ == '__main__':

    SAVE_PATH = '../input/'
    DATASET_PATH = '../input/'
    
    # preprocess train_df_ test_df
    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'public_test.csv'))

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

    train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))
    test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))

    train_df.to_csv(os.path.join(SAVE_PATH, 'preprocessed_train.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(SAVE_PATH, 'preprocessed_test.csv'), index=False, encoding='utf-8')

    # preprocess fasttext vacab
    from gensim.models.wrappers import FastText
    FASTTEXT_PATH = os.path.join(DATASET_PATH, 'fasttext_200.bin')
    model = FastText.load_fasttext_format(FASTTEXT_PATH)
    fasttext = model.wv

    fasttext_vocab = {}

    for word in fasttext.vocab.keys():
        fasttext_vocab[word] = fasttext[word]

    # fasttext_word_list = fasttext_vocab.keys()
    # fasttext_word_vectors_list = fasttext_vocab.values()
    
    import pickle
    with open('../input/fasttext_vocab.pkl', 'wb') as f:
        pickle.dump(fasttext_vocab, f)