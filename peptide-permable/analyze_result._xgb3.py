import pandas as pd
import numpy as np
import os
import xgboost as xgb
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19  }
data = pd.read_csv('chandra_model/Xgb3.csv')
data = pd.read_csv('XGB3_4.csv');data['prob_xgb'] = data['testPred'];data = data[data['len'] ==13]
data['prob'] = data['prob_xgb']
def seq_to_vec(s):
    result = []
    for i in s:
        result += [dictt[i],]
    return np.array(result)
for i in range(13):
    data[i] = 0
data[range(13)]= np.stack(data['seq'].apply(seq_to_vec))
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
#data['var'] = map(np.std, np.array(data[['fold' +str(x) for x in range(0,60)]]))
#data['var'] = data['var']/(59**.5)
best = data.sort_values('prob').reset_index(drop=True)
def get_diff(x):
    return np.argsort(p53_seq_vec != x[[str(y) for y in range(13)]].values)[-3:]
for i in range(1,10):
    print p53_seq
    #print best.iloc[-i][range(0,13)].values, best.iloc[-i].prob,'\n'
    #print np.argsort(p53_seq_vec != best.iloc[-i][range(0,13)].values)[-3:],'\n'
    print best.iloc[-i]['seq'], best.iloc[-i].prob,'\n'
#best['prob'] =  np.log10(best['prob']+0.01)
diff = '0'
def no_K_R(r):
    count =0
    for i in r:
        if i =='R' or i =='K':
            count += 1
    return count

def no_D_E_G(r):
    count =0
    for i in r:
        if i =='D' or i =='E' or i =='G'  :
            count += 1
        elif i =='C'  or i =='M':
            return 9
    return count
if True:
    above_30 = data[data['prob'] >= 0.60]
    score = np.zeros((13,20))
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    for aa in range(0,20):
        for pos in range(0,13):
            score[pos,aa] = np.sum(above_30[above_30[pos] == aa].prob)/np.sum(above_30.prob)

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
            "L" : TextPath((-0.35, 0.003), "L", size=1, prop=fp),
            "N" : TextPath((-0.35, 0), "N", size=1, prop=fp) ,
            "Q" : TextPath((-0.35, 0.01), "Q", size=1, prop=fp),
            "P" : TextPath((-0.35, 0), "P", size=1, prop=fp),
            "S" : TextPath((-0.35, 0.01), "S", size=1, prop=fp),
            "R" : TextPath((-0.35, 0), "R", size=1, prop=fp),
            "T" : TextPath((-0.35, 0), "T", size=1, prop=fp),
            "W" : TextPath((-0.35, 0), "W", size=1, prop=fp),
            "V" : TextPath((-0.35, 0), "V", size=1, prop=fp),
            "Y" : TextPath((-0.35, 0), "Y", size=1, prop=fp) }
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
            for aa in np.argsort(score[i,:]):#for aa in range(0,20)[::-1]:
                temp_score = score[i,aa]
                if temp_score >= thres:
                    letter = dictt_inv[aa]
                    a=letterAt(letter,i+1,y,temp_score,ax)
                    y += temp_score
        plt.xlim((0,14))
        plt.ylim((-0.1,1))
        plt.title(',num samples:'+str(len(above_30)))
        plt.xlabel('peptide position')
        plt.ylabel('probabilities')
        plt.tight_layout()
        plt.xticks(range(1,14),['E1', 'T2', 'F3', 'S4', 'D5', 'L6', 'W7', 'K8', 'L9', 'L10', 'P11', 'E12', 'N13'])
        for i in range(0,13):
            a=letterAt(p53_seq[i],i+1,-0.1,0.09,ax)
        plt.plot((0,14),(0,0),color='black',linewidth='5')
        plt.savefig(name+'.png',dpi=300)
        #plt.show()
        plt.close()
    for i in (5,):
        plot(i*1.0/100,'Fig_60percent%s_thres%s_var'%(diff,i))
    good_no_RK = above_30[above_30['seq'].apply(no_K_R) <= 2].sort_values('prob_xgb')
    below_30 = data[data['prob_xgb'] <= 0.02]
    bad_no_DEG = below_30[below_30['seq'].apply(no_D_E_G) <= 0]
for i in below_30[(below_30['seq'].apply(no_K_R) >= 2) & (below_30['seq'].apply(no_D_E_G) <= 1) ]['seq'].values:
    print i
    print p53_seq
    print '\n',
p53_peptides= [
'ETFSDLWKLLPE',
'TSFAEYWNLLSP',
'LTFEHYWAQLTS',
'LTFEHWWAQLTS',
'LTFEHSWAQLTS',
'ETFEHNWAQLTS',
'LTFEHNWAQLTS',
'LTFEHWWASLTS',
'LTFEHWWSSLTS',
'LTFTHWWAQLTS',
'ETFEHWWAQLTS',
'LTFEHWWSQLTS',
'LTFEHWWAQLLS',
'ETFEHWWSQLLS']

hydropath = np.array([[0.170, 0.500, 0.330, 0.000],
       [-0.240, -0.020, 0.220, 0.000],
       [2.020, 3.630, 1.610, -1.000],
       [1.230, 3.640, 2.410, -1.000],
       [0.010, 1.150, 1.140, 0.000],
       [-1.130, -1.710, -0.580, 0.000],
       [-0.310, -1.120, -0.810, 0.000],
       [0.960, 2.330, 1.370, 0.000],
       [0.990, 2.800, 1.810, 1.000],
       [-0.230, -0.670, -0.440, 0.000],
       [-0.560, -1.250, -0.690, 0.000],
       [0.420, 0.850, 0.430, 0.000],
       [0.580, 0.770, 0.190, 0.000],
       [0.450, 0.140, -0.310, 0.000],
       [0.130, 0.460, 0.330, 0.000],
       [0.810, 1.810, 1.000, 1.000],
       [0.140, 0.250, 0.110, 0.000],
       [-1.850, -2.090, -0.240, 0.000],
       [0.070, -0.460, -0.530, 0.000],
       [-0.940, -0.710, 0.230, 0.000]])
def get_aa_freq(data):
    for i in dictt:
        def fn(str):
            result = 0.0
            for j in str:
                if j.upper() == i:
                    result += 1
            return result
        data['num_'+str(i)] = data['seq'].apply(fn)
        data['per_'+str(i)] = data['num_'+str(i)]/data['len']
    return data
def make_dataframe(p53_peptides):
    len_ = len(p53_peptides[0])
##    for i in p53_peptides:
##            print data_all[data_all['seq']==i]['prob_xgb'],i
    new = pd.DataFrame(p53_peptides,columns=['seq'])
    for i in range(len_):
            new[i] = 0
    new['len']=len_
    new[range(len_)]= np.stack(new['seq'].apply(seq_to_vec))
    seq_tensor = np.transpose(np.eye(20)[new[range(len_)].values.T],(1,2,0))
    num=len(p53_peptides)
    seq_hydropath_global = np.matmul(np.array([hydropath[:,:-4:-1].T,]*num) , seq_tensor)
    combined = np.sum(seq_hydropath_global,2)
    seq_hydropath = np.matmul(np.array([hydropath.T,]*num) , seq_tensor)
    new = get_aa_freq(new)
    for extra in ['length','netcharge','Gwif','Goct']:
        new[extra] = 0
    new['length'] = len_
    new[['netcharge','Gwif','Goct']] = combined[:,0:3]
    xgb_features =['per_A', 'num_A', 'per_C', 'num_C', 'per_E', 'num_E', 'per_D', 'num_D', 'per_G',
                   'num_G', 'per_F', 'num_F', 'per_I', 'num_I', 'per_H', 'num_H', 'per_K', 'num_K',
                   'per_M', 'num_M', 'per_L', 'num_L', 'per_N', 'num_N', 'per_Q', 'num_Q', 'per_P',
                   'num_P', 'per_S', 'num_S', 'per_R', 'num_R', 'per_T', 'num_T', 'per_W', 'num_W',
                   'per_V', 'num_V', 'per_Y', 'num_Y', 'length','netcharge', 'Gwif', 'Goct']
    X = new[xgb_features].copy().values
    xgtest    = xgb.DMatrix(X,missing=np.NAN, feature_names=xgb_features)
    # map(np.array,X) --> it is list of arrays, works that way
    temp_xgb = []
    new['prob_xgb']=0
    xgb_model = []
    for file in sorted([x for x in os.listdir('.') if ('XGB3' in x and '.ckpt' in x)])[::1]:
        bst = xgb.Booster({'nthread':4});bst.load_model(file)
        new['prob'+str(file)]= bst.predict(xgtest)
        temp_xgb += [bst.predict(xgtest),]
        new['prob_xgb'] =  new['prob_xgb'] + temp_xgb[-1]
        xgb_model += ['prob'+str(file),]
    new['var'] = np.var(new[xgb_model],axis=1)
    new['prob_xgb'] = new['prob_xgb']/20
    return new
seq = []
for j in dictt:
	for i in ['LTFIEYWQLLISAA',
	'ATFIEYWQLLISAA',
	'LAFIEYWQLLISAA',
	'LTAIEYWQLLISAA',
	'LTFIAYWQLLISAA',
	'LTFIEAWQLLISAA',
	'LTFIEYAQLLISAA',
	'LTFIEYWALLISAA',
	'LTFIEYWQALISAA',
	'LTFIEYWQLAISAA',
	'LTFIEYWQLLIAAA',]:
		seq += [i[:3]+j+i[4:10]+j+i[11:],]
new = make_dataframe(seq)
astp7041 = ['LTFIEYWQLLISAA',
'ATFIEYWQLLISAA',
'LAFIEYWQLLISAA',
'LTAIEYWQLLISAA',
'LTFIAYWQLLISAA',
'LTFIEAWQLLISAA',
'LTFIEYAQLLISAA',
'LTFIEYWALLISAA',
'LTFIEYWQALISAA',
'LTFIEYWQLAISAA',
'LTFIEYWQLLIAAA',
'LTFMEYWQLLMSAA',
'ATFMEYWQLLMSAA',
'LAFMEYWQLLMSAA',
'LTAMEYWQLLMSAA',
'LTFMAYWQLLMSAA',
'LTFMEAWQLLMSAA',
'LTFMEYAQLLMSAA',
'LTFMEYWALLMSAA',
'LTFMEYWQALMSAA',
'LTFMEYWQLAMSAA',
'LTFMEYWQLLMAAA',]
astp70412=['LTFEYWQLLSAA',
'ATFEYWQLLSAA',
'LAFEYWQLLSAA',
'LTAEYWQLLSAA',
'LTFAYWQLLSAA',
'LTFEAWQLLSAA',
'LTFEYAQLLSAA',
'LTFEYWALLSAA',
'LTFEYWQALSAA',
'LTFEYWQLASAA',
'LTFEYWQLLAAA']
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19  }
