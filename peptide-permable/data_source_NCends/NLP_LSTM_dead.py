import numpy as np
import os
import nltk
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np
import pickle

SENTENCE_LENGTH_MAX = 80
EMBEDDING_DIM=50
#from tensorflow.contrib.keras.api.keras.preprocessing import sequence
from tensorflow.contrib.keras.api.keras.layers import Input, Embedding, GRU, Dense #, Activation
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Bidirectional, TimeDistributed
tokens_input = Input(shape=(SENTENCE_LENGTH_MAX,), dtype='int32', name="SentencesTokens")

sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
data = np.load('file.npy')
data = pd.DataFrame(data,columns=['year', 'date', 'time', 'data', 'site', 'site2', 'date2', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','Change']) 

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
data = data[~data['data_clean'].isin([None,])]

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
    return map(lambda word : 2+word_embedding.dictionary.get(word.lower(),-1),list) #0 for mask 1 for unknown

word_embedding_rnn = np.vstack([ 
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is the 'zero' value (used as a mask in Keras)
        np.zeros( (1, EMBEDDING_DIM,), dtype='float32'),   # This is for 'UNK'  (word == 1)
        word_embedding.word_vectors,
    ])
word_embedding_rnn.shape

data['wordVec'] =  map(word_idx_rnn, data['data_clean_tok'].iloc[:])
data = data[data['wordVec'].apply(len) <= SENTENCE_LENGTH_MAX].reset_index(drop=1)

  
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
tokens_input = Input(shape=(SENTENCE_LENGTH_MAX,), dtype='int32', name="input")

### I want to predict if someone died in the article
ids = []
for i in range(len(data)):
    if 769 in data.iloc[i]['wordVec']:
	    ids += [i,]
print 'confirming 769 is the word dead'  # actually its 767 but 767+2=769 because of keras mask and UNK
print data.iloc[26][['data_clean_tok','wordVec']]
# function to remove 769
def remove769(x):
    result = []
    for i in x:
        if i != 769:
            result += [i,] 
    return result
data['wordVec2'] = data['wordVec'].apply(remove769)
# pad sequences to length 32
data['wordVec2'] = map(lambda x :x + (SENTENCE_LENGTH_MAX - len(x)) * [0,],data['wordVec2'])
data['died'] = 0
data.set_value(ids,'died',1)

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
TAG_SET_SIZE = 1
BATCH_SIZE = 32
RNN_HIDDEN_SIZE = 25
#return sequences retrun full or just inal output
masked = keras.layers.Masking(mask_value=0.0,name='masked') (aggregate_vectors)
rnn_outputs = Bidirectional( GRU(RNN_HIDDEN_SIZE, return_sequences=False),  merge_mode='concat' )(masked)
if True:
##    is_ner_outputs  = TimeDistributed( Dense(3, activation='softmax'), 
##                                       input_shape=(BATCH_SIZE, SENTENCE_LENGTH_MAX, RNN_HIDDEN_SIZE*2),
##                                       name='POS-class')(rnn_outputs)
    rnn_outputs2 = keras.layers.Dropout(0.5,name='drop')(rnn_outputs)
    is_ner_outputs = Dense(1, activation='sigmoid',name='out')(rnn_outputs2)
    model = Model(inputs=[tokens_input], outputs=[is_ner_outputs])
    model.summary()


import keras.backend as K
#https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
def MCC(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy',MCC])
    
X = np.stack(map(np.array,data.wordVec2))
y = np.array(data.died.astype(np.int32))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4, random_state = 5)
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size = 0.5, random_state = 5)
class_weight = {0 : 1.,
                1: 15.7,} # class are ratio 15.7 : 1, helps objective function to be balanced
epochs = 50
history  = model.fit(x=X_train,y=y_train,
                     epochs = epochs,verbose = 2,
                     batch_size = 32,
                     validation_data = (X_val,y_val),
                     class_weight = class_weight)
##for i in range(0,10):
##    model.fit(x=X_train,y=y_train,
##                     epochs = epochs-10,verbose = 2,
##                     batch_size = 320,
##                     validation_data = (X_val,y_val),
##                     class_weight = class_weight)
##    model.save_weights('weights'+str(i)+'.h5')
import matplotlib.pyplot as plt
plt.plot(range(epochs),history.history['MCC'])
plt.plot(range(epochs),history.history['val_MCC'])
#plt.plot(range(epochs),history.history['loss'])
#plt.plot(range(epochs),history.history['val_loss'])
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
