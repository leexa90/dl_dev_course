# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import  numpy as np
import xgboost as xgb
import pandas as pd
import sys
sys.path.append('/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Dropbox/dl_dev_course/peptide-permable/pairwise-alignment-in-python')
import alignment
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19  }
#Interface Scale
#ΔGwif (kcal/mol) 	Octanol Scale
#ΔGwoct (kcal/mol) 	Octanol − Interface
#Scale

dictt_hydropathy  = {
'I' : [ 	-0.31, 	-1.12, 	-0.81 ,0 ],
'L' : [ 	-0.56,	-1.25, 	-0.69 ,0],
'F' : [ 	-1.13, 	-1.71,	-0.58 ,0],
'V' : [ 	0.07 ,	-0.46, 	-0.53 ,0],
'M' : [ 	-0.23, 	-0.67,	-0.44 ,0],
'P' : [ 	0.45 ,	0.14 ,	-0.31 ,0],
'W' : [ 	-1.85, 	-2.09, 	-0.24 ,0],
'T' : [ 	0.14 , 	0.25 , 	0.11 ,0],
'Q' : [ 	0.58 ,	0.77 ,	0.19 ,0],
'C' : [ 	-0.24,	-0.02, 	0.22 ,0],
'Y' : [ 	-0.94, 	-0.71, 	0.23 ,0],
'A' : [ 	0.17 ,	0.50, 	0.33 ,0],
'S' : [ 	0.13 ,	0.46, 	0.33 ,0],
'N' : [ 	0.42 ,	0.85, 	0.43 ,0],
'R' : [ 	0.81 ,	1.81, 	1.00 ,1],
'G' : [ 	0.01 ,	1.15, 	1.14 ,0],
'H' : [ 	0.96 ,	2.33, 	1.37 ,0],
'E' : [ 	2.02 ,	3.63, 	1.61 ,-1],
'K' : [ 	0.99 ,	2.80, 	1.81 ,1],
'D' : [ 	1.23 ,	3.64, 	2.41 ,-1],
'Z' : [ 1.3,  2.2,  0.9, -0.5], #E or Q (mass spec data cannot differentiate)
'B' : [ 0.825,  2.245,  1.42 , -0.5  ], #D or N (mass spec data cannot differentiate)
'X' : [0,0,0,0]} 
data = pd.read_csv('bioactive_PDB_cpp.csv').dropna()
def fn1(str):
    num=0.0
    for i in str:
        if i.upper() == 'R' or i.upper() == 'K':
            num += 1
    if num/len(str) > 0.334: #should be tweaked
        return 0
    return len(set(str))
def fn2(str): # find if sequence has weird bonds
    for i in list(set(str)):
        if i.upper() == '-':
            return 2
    for i in list(set(str)):
        if i.upper() not in dictt:
            return 1    
    return 0
#notable non-cannonical residues , U - selnocysteine
data['type_aa'] = data['seq'].apply(fn1)

data = data[data['type_aa'] >= 4]
print 'removing these number of non-cannonical peptides' ,len(data[data['seq'].apply(fn2) == 1])
data = data[data['seq'].apply(fn2) == 0] # remove peptides with weird chemical bonds , and non-cannonical res (mostly negavtives)
data['len'] = data.seq.apply(len)
data = data.sort_values(by = ['len','source']).reset_index(drop=True)
# get frequencies of amino acids
all = np.concatenate([data.seq])
result = {}
for i in all:
	for j in i :
		if j not in result:
			result[j] =1
		else:
			result[j] += 1
print result
total_aa = np.sum([result[x] for x in result])
for i in result:
	result[i] = np.round(result[i]*1.0 /total_aa,3)
print result
######### Start of initilziing data ###
X = []
counter = 0
## creates num of residue...
features = []
for i in dictt:
    def fn(str):
        result = 0.0
        for j in str:
            if j.upper() == i:
                result += 1
        return result
    data['num_'+str(i)] = data['seq'].apply(fn)
    data['per_'+str(i)] = data['num_'+str(i)]/data['len']
    features += ['per_'+str(i),'num_'+str(i)]

for idx in range(len(data)):
    i = data.iloc[idx]['seq'].upper()
    temp = np.zeros((len(i),5))
    for j in  range(len(i)):
        res = dictt[i[j]]
        temp[j][0] = res
        #temp[j][-1] = len(i)*0.01
        temp[j][-4:] = dictt_hydropathy[i[j]]
    temp = temp.T
    alternative = [len(i),np.sum(temp[-1,:]),np.sum(temp[-2,:]),np.sum(temp[-3,:])]
    alternative += list(data.iloc[idx][features]) #add ratio for aa
    per = data.iloc[idx]['source'] == 2 
    X += [[temp[0],temp[1:],alternative,i,per*1],]
    #print '> %s\n%s' %(i,i)
    counter += 1
data['X'] = X
kmean = np.load('Kmean.npy')
data['fold']=kmean
data = data.sort_values(by = ['len','source']).reset_index(drop=True)
for i in ['length','netcharge','Gwif','Goct']:
    data[i] = 0
data['y'] = 0
for i in range(0,len(data)):
    temp = data.iloc[i]['X'][-3]
    zz = data.set_value(i,['length','netcharge','Gwif','Goct'],temp[0:4])
    zz = data.set_value(i,'y',data.iloc[i]['X'][-1])

features=['per_A', 'num_A', 'per_C', 'num_C', 'per_E', 'num_E', 'per_D', 'num_D', 'per_G', 'num_G',
 'per_F', 'num_F', 'per_I', 'num_I', 'per_H', 'num_H', 'per_K', 'num_K', 'per_M', 'num_M',
 'per_L', 'num_L', 'per_N', 'num_N', 'per_Q', 'num_Q', 'per_P', 'num_P', 'per_S', 'num_S', 'per_R', 'num_R', 'per_T', 'num_T',
 'per_W', 'num_W', 'per_V', 'num_V', 'per_Y', 'num_Y', 'length', 'netcharge', 'Gwif', 'Goct']

X = data[features+['y']].copy().values
import sklearn.metrics,random

sim_matrix = np.zeros((len(data),len(data)))

import sys
if len(sys.argv) >= 2:
    start = sys.argv[1]
else:
    start = 0
print start
start = int(start)
if False:
    for i in range(start,len(data),7):
        seq1 = data.iloc[i]['seq']
        print seq1
        for j in range(i+1,len(data)):
            seq2 = data.iloc[j]['seq']
            score = alignment.needle(seq1,seq2)[-1]
            sim_matrix[i,j] = score
            sim_matrix[j,i] = score
    np.savez_compressed('sim_matrix%s.npy.zip'%start,sim_matrix)
if True:
    for start in range(0,7):
         sim_matrix =  sim_matrix + np.load('sim_matrix%s.npy.zip.npz' %start).items()[0][1]
def swap_rows(C, var1, var2):
        '''
        Function to swap two rows in a covariance matrix,
        updating the appropriate columns as well.
        '''
        D = C.copy()
        D[var2, :] = C[var1, :]
        D[var1, :] = C[var2, :]

        E = D.copy()
        E[:, var2] = D[:, var1]
        E[:, var1] = D[:, var2]

        return E
die
if True: 
    from sklearn.cluster import KMeans
    eigen_values, eigen_vectors = np.linalg.eigh(sim_matrix)
    means=KMeans(n_clusters=5, init='k-means++').fit_predict(eigen_vectors[:, -2:])
dictt = {0: 'ro', 1: 'bo', 2: 'go', 3: 'ko', 4: 'co'}
for mean in range(0,5):
    x = []
    y = []
    for i in np.matmul(sim_matrix[means==mean],eigen_vectors[:, -2:]):
        x += [i[0],]
        y += [i[1],]
    plt.plot(x,y,dictt[mean])
plt.show()
        

                
            
#https://stats.stackexchange.com/questions/138325/clustering-a-correlation-matrix
if False:
    import numpy as np
    from matplotlib import pyplot as plt

    # This generates 100 variables that could possibly be assigned to 5 clusters
    n_variables = 9727
    n_clusters = 5
    n_samples = 1000

    # To keep this example simple, each cluster will have a fixed size
    cluster_size = n_variables / n_clusters

    # Assign each variable to a cluster
    belongs_to_cluster = np.repeat(range(n_clusters), cluster_size)
    np.random.shuffle(belongs_to_cluster)

    # This latent data is used to make variables that belong
    # to the same cluster correlated.
    latent = np.random.randn(n_clusters, n_samples)

##    variables = []
##    for i in range(n_variables):
##        variables.append(
##            np.random.randn(n_samples) + latent[belongs_to_cluster[i], :]
##        )
##
##    variables = np.array(variables)

    C = sim_matrix

    def score(C):
        '''
        Function to assign a score to an ordered covariance matrix.
        High correlations within a cluster improve the score.
        High correlations between clusters decease the score.
        '''
        score = 0
        for cluster in range(n_clusters):
            inside_cluster = np.arange(cluster_size) + cluster * cluster_size
            outside_cluster = np.setdiff1d(range(n_variables), inside_cluster)

            # Belonging to the same cluster
            score += np.sum(C[inside_cluster, :][:, inside_cluster])

            # Belonging to different clusters
            score -= np.sum(C[inside_cluster, :][:, outside_cluster])
            score -= np.sum(C[outside_cluster, :][:, inside_cluster])

        return score


    initial_C = C
    initial_score = score(C)
    initial_ordering = np.arange(n_variables)

    plt.figure()
    plt.imshow(C, interpolation='nearest')
    plt.title('Initial C')
    print 'Initial ordering:', initial_ordering
    print 'Initial covariance matrix score:', initial_score

    # Pretty dumb greedy optimization algorithm that continuously
    # swaps rows to improve the score
    def swap_rows(C, var1, var2):
        '''
        Function to swap two rows in a covariance matrix,
        updating the appropriate columns as well.
        '''
        D = C.copy()
        D[var2, :] = C[var1, :]
        D[var1, :] = C[var2, :]

        E = D.copy()
        E[:, var2] = D[:, var1]
        E[:, var1] = D[:, var2]

        return E

    current_C = C
    current_ordering = initial_ordering
    current_score = initial_score

    max_iter = 100
    for i in range(max_iter):
        print i,
        # Find the best row swap to make
        best_C = current_C
        best_ordering = current_ordering
        best_score = current_score
        for row1 in range(np.random.randint(0,20),n_variables,20):
            for row2 in range(row1+np.random.randint(0,20),n_variables,20):
                if row1 == row2:
                    continue
                option_ordering = best_ordering.copy()
                option_ordering[row1] = best_ordering[row2]
                option_ordering[row2] = best_ordering[row1]
                option_C = swap_rows(best_C, row1, row2)
                option_score = score(option_C)

                if option_score > best_score:
                    best_C = option_C
                    best_ordering = option_ordering
                    best_score = option_score

        if best_score > current_score:
            # Perform the best row swap
            current_C = best_C
            current_ordering = best_ordering
            current_score = best_score
            print score
        else:
            # No row swap found that improves the solution, we're done
            break

    # Output the result
    plt.figure()
    plt.imshow(current_C, interpolation='nearest')
    plt.title('Best C')
    print 'Best ordering:', current_ordering
    print 'Best score:', current_score
    print
    print 'Cluster     [variables assigned to this cluster]'
    print '------------------------------------------------'
    for cluster in range(n_clusters):
        print 'Cluster %02d  %s' % (cluster + 1, current_ordering[cluster*cluster_size:(cluster+1)*cluster_size])
        
kill

RESULT = {}
import sys
if len(sys.argv)==2:
    test = int(sys.argv[1])
else:
    test = 2
print "TEST IS",test
all_data =[]

def get_data_from_X(X,y,i): #get tensor inputs from X and y
    Inp0_ = np.array([X[i][0]])
    Inp1_ = np.array([X[i][1]])
    Inp2_ = np.array([X[i][2]])
    labels_ = np.array([[y[i],]])
    return Inp0_,Inp1_,Inp2_,labels_
folds= 5
for test in range(0,5):
    X = data[features+['y']].copy().values
    folds= 5
    # ensure no leakage of train to test, was getting 95 AUC
    X_test = []
    y_test = []
    print 'train_val_test size:' , len(X)
    for i in range(test,len(X),folds):
        if i%5 == test:
            x = X[i].copy() #or else x is only a reference
            X_test += [x[:-1],]
            y_test += [x[-1],]
            X[i] = -999
    X = [x for x in X if -999 not in x]
    print 'train+val  size :', len(X)
    test_emsemble= []

    for repeat in range(0,1): #perform 5 repeats
        for CV in range(folds-1): #for each repeat, do 4 fold CV. (test set is kept constant throughtout)# 
            RESULT[CV] = []
            X_train = []
            y_train = []
            X_val = []
            y_val = []    
            for i in range(len(X)):
                x = X[i]
                if i%5 == CV:
                    X_val += [x[:-1],]
                    y_val += [x[-1],]
                else:
                    X_train += [x[:-1],]
                    y_train += [x[-1],]
            predictors = features
            xgcv    = xgb.DMatrix(X_val, label=y_val,missing=np.NAN,feature_names=predictors)
            print predictors
            xgtest    = xgb.DMatrix(X_test, label=y_test,missing=np.NAN,feature_names=predictors)
            xgtrain = xgb.DMatrix(X_train, label=y_train,missing=np.NAN,feature_names=predictors)
            watchlist  = [ (xgtrain,'train'),(xgtest,'test'),(xgcv,'eval')]
            a = {}
            params = {}
            params["objective"] =  "binary:logistic"
            params["eta"] = 0.01
            params["min_child_weight"] = 2
            params["subsample"] = 0.7
            params["colsample_bytree"] = 0.7
            params["scale_pos_weight"] = 1.0
            params["silent"] = 500
            params["max_depth"] = 6
            params['seed']=1
            #params['maximize'] =True
            params['eval_metric'] =  'auc'
            plst = list(params.items())
            early_stopping_rounds=200
            best_roc_val = {}     # stores val ROC for trainnig epochs per CV+repeat run
            for epoch in range(1):#training_epochs):
                result_d=xgb.train(plst,xgtrain,2200,watchlist,
                                   early_stopping_rounds=early_stopping_rounds,
                                   evals_result=a,maximize=0,verbose_eval=500)
                print result_d.get_fscore()
                logit_train = []
                pred = result_d.predict(xgtrain)
                roc_train = sklearn.metrics.roc_auc_score(y_train,pred)
                pred = result_d.predict(xgcv)
                roc_val = sklearn.metrics.roc_auc_score(y_val,pred)
                pred = result_d.predict(xgtest)
                roc_test = sklearn.metrics.roc_auc_score(y_test,pred)

                print roc_train,roc_val,roc_test
                all_data += [[roc_train,roc_val,roc_test],]

                test_emsemble += [result_d.predict(xgtest),]
                model_name = 'XGB1_%s_%s_%s_%s_%s_%s.ckpt' %(test,CV,repeat,str(roc_train)[:5],str(roc_val)[:5],str(roc_test)[:5])
                result_d.save_model(model_name)
                del result_d
            print sklearn.metrics.roc_auc_score(y_test,np.mean(np.array(test_emsemble),0))

    zz=data.set_value(range(test,len(data),5),'testPred',np.mean(np.array(test_emsemble),0))
    zz=data.set_value(range(test,len(data),5),'testY',y_test)
    data.iloc[range(test,len(data),5)].to_csv('XGB_%s.csv' %test,index=0)
print sklearn.metrics.roc_auc_score(data.testY,data.testPred)
data.to_csv('XGB_all.csv',index=0)
