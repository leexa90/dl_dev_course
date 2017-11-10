import pandas as pd

neg = []
for line in open('peptide_bioactive','r'): #signalling from all 
    if '\n' in line:
        line = line[:-1]
    neg += [line,]
def smaller_than_31(x):
    if len(x) <= 30:
        return x
neg = map(smaller_than_31,neg)
neg_seq = list(set(neg))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=30);#plt.show()
neg1 = neg_seq
neg = pd.read_csv('peptide.csv') #pDB
neg = neg[neg.Sequence.apply(len) < 31]
neg_seq = list(set(neg.Sequence))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=28);#plt.show()
neg2 = neg_seq

neg = pd.read_csv('Mammal_signal_peptide.csv') #
neg = neg[neg.seq.apply(len) < 31]
neg_seq = list(set(neg.seq))
plt.hist(map(len,neg_seq),bins=28);#plt.show()
neg3 = neg_seq

neg = []
for line in open('peptide2.csv','r'): #CPP2
    if '\n' in line:
        line = line[:-1]
    if '>' not in line:
        neg += [line,]
def smaller_than_31(x):
    if len(x) <= 30:
        return x
neg = [x for x in map(smaller_than_31,neg) if x is not None]
neg_seq = list(set(neg))
import matplotlib.pyplot as plt
plt.hist(map(len,neg_seq),bins=30);#plt.show()

al = neg1+neg2+neg3+neg_seq
data= pd.DataFrame(al, columns=['seq'])
data['source'] = [0,]*len(neg1) + [1,]*len(neg2) + [-1,]*len(neg3) + [2,]*len(neg_seq)
data = data.dropna()
data['size'] = data['seq'].apply(len)
data = data.sort_values(['size','seq','source'])
for i in range(1,len(data)):
	if data.iloc[i].seq.upper() == data.iloc[i-1].seq.upper():
		print data.iloc[i-1:i+1]
		zz=data.set_value(i-1,'size' ,-1)
data  = data[data['size']  >= 4]
data.to_csv('bioactive_PDB_cpp.csv',index=0)
