import re
import os
import numpy as np
import pandas as pd
from collections import Counter
from .utils import ON_KAGGLE 
if not ON_KAGGLE:
    from soynlp.hangle import decompose



def build_matrix(word_index, word2vec_vocab, max_features, vector_size=200):
    embedding_matrix = np.zeros((max_features + 1, vector_size))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = word2vec_vocab[word]
            except:
                unknown_words.append(word)
                
    return embedding_matrix, unknown_words

def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
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

    # train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))
    # test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))

    # train_df.to_csv(os.path.join(SAVE_PATH, 'preprocessed_train.csv'), index=False, encoding='utf-8')
    # test_df.to_csv(os.path.join(SAVE_PATH, 'preprocessed_test.csv'), index=False, encoding='utf-8')

    # preprocess fasttext vacab
    from gensim.models.wrappers import FastText
    FASTTEXT_PATH = os.path.join(DATASET_PATH, '4gram_model_file.bin')
    model = FastText.load_fasttext_format(FASTTEXT_PATH)
    fasttext = model.wv

    fasttext_vocab = {}

    for word in fasttext.vocab.keys():
        fasttext_vocab[word] = fasttext[word]

    # fasttext_word_list = fasttext_vocab.keys()
    # fasttext_word_vectors_list = fasttext_vocab.values()
    
    import pickle
    with open('../input/fasttext_4gram_vocab.pkl', 'wb') as f:
        pickle.dump(fasttext_vocab, f)