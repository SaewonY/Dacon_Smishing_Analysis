!pip install soynlp
!pip install fasttext

##################################################


from soynlp.hangle import decompose

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
def jamo_sentence(sent):
    sent = sent.replace('XXX', 'X')
    def transform(char):
        if char == ' ':
            return char
        elif char == 'X':
            return ' X'
        elif char == '.':
            return '. '
        else:
            return char

    sent_ = ''.join(transform(char) for char in sent)
    sent_ = doublespace_pattern.sub(' ', sent_)
    return sent_

jamo_sentence(train_df.loc[0, 'text'])



##################################################
test_df['jamo'] = test_df['text'].apply(lambda x: jamo_sentence(x))
train_df['jamo'] = train_df['text'].apply(lambda x: jamo_sentence(x))


##################################################
raw_corpus_fname = '\n'.join(pd.concat([train_df['jamo'], test_df['jamo']]))
file=open('/content/textfile.txt','w')
file.write(raw_corpus_fname)
file.close()
##################################################

import fasttext
model = fasttext.train_unsupervised('/content/textfile.txt', 
                                    model='cbow' ,
                                    loss = 'hs',        # hinge loss
                                    ws=1,               # window size
                                    lr = 0.01,          # learning rate
                                    dim = 200,          # embedding dimension
                                    epoch = 5,          # num of epochs
                                    min_count = 10,     # minimum count of subwords
                                    # encoding = 'utf-8', # input file encoding
                                    thread = -1          # num of threads
                                )


##################################################

model.save_model("/content/model_file.bin")
model.save_model("/content/drive/My Drive/model_file.bin")
