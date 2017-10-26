import numpy as np
import os
import nltk
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np
import pickle

SENTENCE_LENGTH_MAX = 32
EMBEDDING_DIM=50



#from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from tensorflow.contrib.keras.api.keras.layers import Input, Embedding, GRU, Dense #, Activation
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Bidirectional, TimeDistributed
tokens_input = Input(shape=(SENTENCE_LENGTH_MAX,), dtype='int32', name="SentencesTokens")

sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
data = np.load('file.npy')
data = pd.DataFrame(data,columns=['year', 'date', 'time', 'data', 'site', 'site2', 'date2', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']) 
import string
printable = set(string.printable)
def remove_nonprintable(s):
    if s is None:
        return None
    return filter(lambda x: x in printable, s)

from nltk.tokenize import TreebankWordTokenizer
data['data_clean'] = data['data'].apply(remove_nonprintable)
def tokenizer(s):
    if s is None:
        return None
    return TreebankWordTokenizer().tokenize(s)
data['data_clean_tok'] = data['data_clean'].apply(tokenizer)

import sys
sys.path.append('/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/deep-learning-workshop/notebooks/5-RNN/glove-python')
import glove
glove_dir = './data/RNN/'
glove_100k_50d = 'glove.first-100k.6B.50d.txt'
glove_100k_50d_path = os.path.join(glove_dir, glove_100k_50d)
if not os.path.isfile( glove_100k_50d_path ):
    raise RuntimeError("You need to download GloVE Embeddings "+
                       ": Use the downloader in 5-Text-Corpus-and-Embeddings.ipynb")
else:
    print("GloVE available locally")
word_embedding = glove.Glove.load_stanford( glove_100k_50d_path )
word_embedding.word_vectors.shape
def word_idx_rnn(list):
    if list is None:
        return None
    return map(lambda word : word_embedding.dictionary.get(word.lower(),-1),list)

word_embedding_rnn = np.vstack([ 
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is the 'zero' value (used as a mask in Keras)
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is for 'UNK'  (word == 1)
        word_embedding.word_vectors,
    ])
word_embedding_rnn.shape

data['wordVec'] =  map(word_idx_rnn, data['data_clean_tok'].iloc[:])
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
tokens_input = Input(shape=(SENTENCE_LENGTH_MAX,), dtype='int32', name="SentencesTokens")

# load pre-trained word embeddings into an Embedding layer
#   note that we set trainable = False so as to keep the embeddings fixed
embedded_sequences = Embedding(word_embedding_rnn.shape[0],
                                EMBEDDING_DIM,
                                weights=[ word_embedding_rnn ],
                                input_length=SENTENCE_LENGTH_MAX,
                                trainable=False, 
                                mask_zero=True,
                                name="SentencesEmbedded") (tokens_input)

#extra_input = ...
aggregate_vectors = embedded_sequences # concat...

rnn_outputs = Bidirectional( GRU(49, return_sequences=True),  merge_mode='concat' )(aggregate_vectors)

is_ner_outputs  = TimeDistributed( Dense(TAG_SET_SIZE, activation='softmax'), 
                                   input_shape=(BATCH_SIZE, SENTENCE_LENGTH_MAX, RNN_HIDDEN_SIZE*2),
                                   name='POS-class')(rnn_outputs)
######################
die 
df2 = pd.read_csv('ES3.SI.csv')
print df2
from datetime import date as DATE
import datetime
for i in range(1,len(df2)):
    before = map( np.int, df2.iloc[i-1]['Date'].split('-'))
    current = map( np.int, df2.iloc[i]['Date'].split('-'))
    if DATE(current[0],current[1],current[2]) - DATE(before[0],before[1],before[2])\
    != datetime.timedelta(1):
        x = DATE(current[0],current[1],current[2]) - DATE(before[0],before[1],before[2])
        for ii in range(1,x.days):
            newdate = DATE(before[0],before[1],before[2]) + datetime.timedelta(ii)
            newdate = newdate.isoformat()
            temp = [newdate,]+ list(df2.iloc[i-1] [df2.keys()[1:]],)
            df2 = df2.append(pd.DataFrame([temp ], columns= list(df2.keys())))
    

df = df2.reset_index(drop=1)

def seperate(j):
    result = []
    prev = 0
    for i in range(len(j)):
                    if j[i] == ':':
                            if (j[i-2:i]+j[i+1:i+3]).isnumeric():
                                    result +=[ j[prev:i-2],]
                                    prev = i-2
    result += [j[prev:],]
    return result
def get_link(j):
    result = []
    for i in range(len(j)):
        if j[i:i+14]=='http://str.sg/':
            result=[j[:i],j[i:].split()[0],j[i:].split()]
    return result

df = pd.DataFrame([[np.nan,]*6],columns=['year','date','time','data','site','site2'])
for year in ['2016','2017']:
    for file in [x for x in os.listdir('.') if year+'.txt' in x]:
        dictt={}
        for line in open(file,'r'):
            #print line
            #x =  sentence_splitter.tokenize(line.decode('utf-8'))
            if  '.txt' in line:
                line = line[:-1]
                dictt[line] = []
                impt = line
            else:
                dictt[impt] =seperate(line.decode('utf-8'))[2:]
            for a in seperate(line.decode('utf-8'))[:]:
                None#print a

        for date in dictt:
            temp = []
            for article in dictt[date]:
                #print article
                temp +=[ [year,date,article[0:5],]+get_link(article[5:]),]
            df = df.append(pd.DataFrame(temp,columns=['year','date','time','data','site','site2']))
df = df.reset_index(drop=1)   



dictt_month = {'march': '03', 'august': '08', 'may': '05', 'june': '06', 'jan': '01', 'april': '04',
               'feb': '02', 'july': '07', 'dec': '12', 'sept': '09', 'oct': '10', 'nov': '11'}

def fun(x):
    result =  [dictt_month[x[:-4].split('_')[0]],x[:-4].split('_')[1]]
    if len(result[1]) == 1:
        result[1] = '0' + result[1]
    return result

df['date'][1:].apply(fun)
df = df.iloc[1:]
df['date2'] = df['date'][:].apply(fun)
df['Date'] = map(lambda x :x[1]['year']+'-'+ x[1]['date2'][0]+'-'+ x[1]['date2'][1],df.iloc[:].iterrows())
df_comb = pd.merge(df,df2,'left',on='Date')
np.save('file.npy',np.array(df_comb))
df_comb.to_csv('Data.csv')
