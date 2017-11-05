import pandas as pd

import os

files = sorted([x for x in os.listdir('results') if '.csv' in x])
dict_files = {}
data = pd.read_csv('results/'+files[0])
data['diff'] = 0
dict_files[0] = files[0]
counter =1
for i in files[1:]:
    print i
    counter += 1
    dict_files[counter] = i
    temp = pd.read_csv('results/'+i)
    temp['diff'] = counter
    data = pd.concat([data,temp])

import numpy as np
import matplotlib.pyplot as plt
size= np.log10(data.prob)
plt.hist(size,bins=100)

dictt_inv = {0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K', 9: 'M',
             10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y'}

def string(arr):
    result = ''
    for i in arr:
        result += dictt_inv[i]
    return result

p53_seq='ETFSDLWKLLPEN'
p53_seq_vec = np.array([2., 16., 5., 14., 3., 10., 17., 8., 10., 10., 13., 2., 11.])
data['var'] = map(np.std, np.array(data[data.keys()[14:-1]]))
best = data.sort_values('prob')[list(data.keys()[0:14])+['diff',]].reset_index(drop=True)
def get_diff(x):
    return np.argsort(p53_seq_vec != x[[str(y) for y in range(13)]].values)[-3:]
for i in range(1,10):
    #print p53_seq_vec
    #print best.iloc[-i][range(0,13)].values, best.iloc[-i].prob,'\n'
    print np.argsort(p53_seq_vec != best.iloc[-i][range(0,13)].values)[-3:],'\n'
    print string(best.iloc[-i][range(0,13)].values), best.iloc[-i].prob,'\n'
#best['prob'] =  np.log10(best['prob']+0.01)
above_30 = data[data['prob']-data['var'] >= 0.3]
score = np.zeros((13,20))
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
for aa in range(0,20):
    for pos in range(0,13):
        score[pos,aa] = np.sum(above_30[above_30[str(pos)] == aa].prob)/np.sum(above_30.prob)

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

fp = FontProperties(family="monospace", weight="bold") 
globscale = 1.35
LETTERS = {
        "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C" : TextPath((-0.35, 0), "C", size=1, prop=fp),
        "E" : TextPath((-0.35, 0), "E", size=1, prop=fp),
        "D" : TextPath((-0.35, 0), "D", size=1, prop=fp) ,
         "G" : TextPath((-0.35, 0), "G", size=1, prop=fp),
        "F" : TextPath((-0.35, 0), "F", size=1, prop=fp),
        "I" : TextPath((-0.35, 0), "I", size=1, prop=fp),
        "H" : TextPath((-0.35, 0), "H", size=1, prop=fp) ,
        "K" : TextPath((-0.35, 0), "K", size=1, prop=fp),
        "M" : TextPath((-0.35, 0), "M", size=1, prop=fp),
        "L" : TextPath((-0.35, 0), "L", size=1, prop=fp),
        "N" : TextPath((-0.35, 0), "N", size=1, prop=fp) ,
        "Q" : TextPath((-0.35, 0), "Q", size=1, prop=fp),
        "P" : TextPath((-0.35, 0), "P", size=1, prop=fp),
        "S" : TextPath((-0.35, 0), "S", size=1, prop=fp),
        "R" : TextPath((-0.35, 0), "R", size=1, prop=fp),
        "T" : TextPath((-0.35, 0), "T", size=1, prop=fp),
        "W" : TextPath((-0.35, 0), "W", size=1, prop=fp),
        "V" : TextPath((-0.35, 0), "V", size=1, prop=fp),
        "Y" : TextPath((-0.35, 0), "Ys", size=1, prop=fp) }
COLOR_SCHEME = {'A': 'grey', 'C': 'lightBlue', 'E': 'red', 'D': 'red',
                'G': 'grey', 'F': 'green', 'I': 'grey', 'H': 'blue', 'K': 'blue',
                'M': 'grey', 'L': 'grey', 'N': 'lightBlue', 'Q': 'lightBlue', 'P': 'orange',
                'S': 'lightBlue', 'R': 'blue', 'T': 'lightBlue', 'W': 'green', 'V': 'grey',
                'Y': 'green'}


def letterAt(letter, x, y, yscale=1, ax=None):

    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def plot(thres=0.05,name='temp'):
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(0,13):
        y = 0
        for aa in range(0,20)[::-1]:
            temp_score = score[i,aa]
            if temp_score >= thres:
                letter = dictt_inv[aa]
                a=letterAt(letter,i+1,y,temp_score,ax)
                y += temp_score
    plt.xlim((0,14))
    plt.ylim((0,1))
    plt.tight_layout()
    plt.xticks(range(1,14),['E1', 'T2', 'F3', 'S4', 'D5', 'L6', 'W7', 'K8', 'L9', 'L10', 'P11', 'E12', 'N13'])
    plt.savefig(name+'.png',dpi=300)
    plt.show()
    plt.close()
plot(0.05,'Fig_30percent_thres5_var')
