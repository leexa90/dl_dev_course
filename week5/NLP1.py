import numpy as np
import os
import nltk
import pandas as pd
#nltk.download('punkt')
sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

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
    for file in [x for x in os.listdir('.') if '2016.txt' in x]:
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
die
df2 = pd.read_csv('ES3.SI.csv')

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
