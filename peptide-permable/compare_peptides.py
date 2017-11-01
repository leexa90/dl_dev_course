import pandas as pd

neg = []
for line in open('peptide_bioactive','r'):
    if '\n' in line:
        line = line[:-1]
    neg += [line,]
def smaller_than_31(x):
    if len(x) <= 31:
        return x
neg = map(smaller_than_31,neg)
neg_seq = list(set(neg))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=30);#plt.show()
neg1 = neg_seq
neg = pd.read_csv('peptide.csv')
neg = neg[neg.Sequence.apply(len) < 31]
neg_seq = list(set(neg.Sequence))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=28);#plt.show()
neg2 = neg_seq

neg = []
for line in open('peptide2.csv','r'):
    if '\n' in line:
        line = line[:-1]
    if '>' not in line:
        neg += [line,]
def smaller_than_31(x):
    if len(x) <= 31:
        return x
neg = [x for x in map(smaller_than_31,neg) if x is not None]
neg_seq = list(set(neg))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=30);#plt.show()

al = neg1+neg2+neg_seq
data= pd.DataFrame(al, columns=['seq'])
data['source'] = [0,]*len(neg1) + [1,]*len(neg2) + [2,]*len(neg_seq)
data = data.dropna()
data.to_csv('bioactive_PDB_cpp.csv',index=0)
