# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import  numpy as np
import tensorflow as tf
import pandas as pd
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
    if num/len(str) > 0.333: #should be tweaked
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
import sys
sys.path.append('pairwise-alignment-in-python/')
import alignment
p53= 'ETFSDLWKLLPEN'

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

def make(seq,per=0):
        i = seq
        temp = np.zeros((len(i),5))
        for j in  range(len(i)):
            res = dictt[i[j]]
            temp[j][0] = res
            #temp[j][-1] = len(i)*0.01
            temp[j][-4:] = dictt_hydropathy[i[j]]
        alternative = [len(i),np.sum(temp[-1]),np.sum(temp[-2]),np.sum(temp[-3])]
        per = per
        temp = temp.T
        X = [temp[0],temp[1:],alternative,i,per*1]
        #print '> %s\n%s' %(i,i)
        return X

result = []
for i in data.seq:
    result += [ alignment.needle(i,p53)[-1],]
data['1']= result
data.sort_values('1')
p53 = make(p53)

print len(result)


epsilon = 1e-3
def batch_normalization(x,name='batchnorm',feature_norm = False):
    # ideally i want to do batch norm per row per sample
    #epsilon = tf.Variable(tf.constant([1e-3,]*x.shape[0])) 
    if feature_norm : 
##        mean,var = tf.nn.moments(x,[2,3],keep_dims=True)
##        scale = tf.Variable(tf.ones([1,x.shape[1],1,1]))
##        beta = tf.Variable(tf.zeros([1,x.shape[1],1,1]))
##        x = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon,name=name)
        x = tf.contrib.layers.layer_norm (x,trainable=False)
    else:
        x = tf.contrib.layers.layer_norm (x,trainable=False)
##        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)
##        scale = tf.Variable(tf.ones([1,1,1,x.shape[-1]]))
##        beta = tf.Variable(tf.zeros([1,1,1,x.shape[-1]]))
##        x = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon,name=name)
    return x


learning_rate = 0.00001
training_epochs = 100
batch_size = 1
display_step = 1
n_classes =5
with tf.name_scope('inputs') as scope:
    Inp0 = tf.placeholder(tf.int32,[None,None],name='sequence_factors1')
    Inp1 = tf.placeholder(tf.float32,[None,4,None],name='sequence_factors2')
    labels = tf.placeholder(tf.float32 , [None,1],name='labels')
    dropout = tf.placeholder(tf.float32,name='dropout')
    Inp2 = tf.placeholder(tf.float32, [None,4] ,name = 'globa_seq_info')

with tf.name_scope('embedding') as scope:
    aa_embeddings = tf.get_variable('aa_embeddings',[20, 5])
    embedded_word_ids = tf.gather(aa_embeddings,range(0,20))
    embed0 = tf.nn.embedding_lookup(aa_embeddings,Inp0,name='lookup')
    embed1 = tf.transpose(embed0,(0,2,1))
    unstack0 = tf.unstack(Inp1,axis=-2,name='unstack0')
    unstack1 = tf.unstack(embed1 , axis=-2,name='unstack1')
    layer0 = tf.stack([tf.stack(unstack0+unstack1,axis=1)],-1,name='stack')

with tf.name_scope('layer1') as scope:
    layer1_norm = batch_normalization(layer0,'BN_layer0',feature_norm = True)
    layer1 = tf.layers.conv2d(layer1_norm,32,(9,1),padding='valid',activation=tf.nn.relu)
    layer1_DO = tf.layers.dropout(layer1,rate=dropout,name='Drop1')

with tf.name_scope('layer2') as scope:
    layer2_norm = batch_normalization(layer1_DO,'BN_layer1',feature_norm = True)
    layer2 = tf.layers.conv2d(layer2_norm,64,(1,3),padding='same',activation=tf.nn.relu)
    layer2_DO = tf.layers.dropout(layer2,rate=dropout,name='Drop2')

with tf.name_scope('layer3') as scope:
    layer3_norm = batch_normalization(layer2_DO,'BN_layer1')
    layer3 = tf.layers.conv2d(layer3_norm,96,(1,3),padding='same',activation=tf.nn.relu)
    layer3_DO = tf.layers.dropout(layer3,rate=dropout,name='Drop3')

with tf.name_scope('layer4') as scope:
    layer4_norm = batch_normalization(layer3_DO,'BN_layer2')
    layer4 = tf.layers.conv2d(layer4_norm,128,(1,3),padding='same',activation=tf.nn.relu)
    layer4_DO = tf.layers.dropout(layer4,rate=dropout,name='Drop4')

with tf.name_scope('dense') as scope:
    globalmeanpooling = tf.reduce_mean(layer4[:,:,:,0::2],(1,2),name='globalmaxpooling')
    globalmaxpooling = tf.reduce_mean(layer4[:,:,:,1::2],(1,2),name='globalmaxpooling')
    gbmp_extra = tf.concat([Inp2,globalmaxpooling,globalmeanpooling],axis = 1, name ='gbmp_extra')
    layer5_DO = tf.layers.dropout(gbmp_extra,rate=dropout,name='Drop5')
    dense1 = tf.layers.dense(layer4_DO,16,activation = tf.nn.relu , name = 'dense1' )
    layer6_DO = tf.layers.dropout(dense1,rate=dropout,name='Drop6')
    dense2 = tf.layers.dense(globalmaxpooling,1 , name = 'dense2' )
with tf.name_scope('loss') as scope:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                      logits = dense2,
                                                      name='loss')
with tf.name_scope('output') as scope:
    out_softmax = tf.nn.sigmoid(dense2)
#learning_rate = tf.Variable(0,dtype= np.float32)
mean_loss = tf.reduce_mean(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')

with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(mean_loss)
with tf.name_scope('accuracy') as scope:
    predict_boo = tf.greater(out_softmax,0.5)
    predict = tf.cast(predict_boo, np.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels,predict),tf.float32),name='accuracy')
import sklearn.metrics,random
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters)


RESULT = {}


all_data =[]
test_emsemble= []
def get_data_from_X(X,y,i=i): #get tensor inputs from X and y
    Inp0_ = np.array([X[i][0]])
    Inp1_ = np.array([X[i][1]])
    Inp2_ = np.array([X[i][2]])
    labels_ = np.array([[y[i],]])
    return Inp0_,Inp1_,Inp2_,labels_
folds= 5
lr = 0
XX = [(p53[0],p53[1],p53[2]),]
yy = [p53[-1],]
Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(XX,yy,0)


saver.restore(sess,'temp_models/model2_2_0_0_0.908_0.849_0.872.ckpt')

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
dictt_inv = {0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K', 9: 'M',
             10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y'}

def mutate(seq,unchanged=(2,6,9)):
    new = np.random.randint(0,20,size=(len(seq),10))
    X_0 = new
    new = np.transpose(np.eye(20)[new],(1,2,0))
    temp = np.matmul(np.array([hydropath.T,]*10) , new)
    length = np.array([[len(seq)],]*10)
    alternative = np.sum(temp[:,1:,:],2)
    X_2 = np.concatenate((length,alternative),1)
    X_1 = temp
    return X_0,X_1,X_2
def mutate(seq,unchanged=(2,6,9)):
    resn = unchanged[0]
    while resn in unchanged:
        resn = np.random.randint(0,len(seq))
    aa = np.random.randint(0,20)
    seq = seq[0:resn] + dictt_inv[aa] + seq[resn+1:]
    return seq

def mutate(seq,resnum,aa,unchanged=(2,6,9)):
    for i,j in zip(resnum,aa):
        if i not in unchanged:
            seq = seq[0:i]+  dictt_inv[j] +seq[i+1:]
    return seq

def make_batch400(seq,c1,c2,per=0):
        assert (c1 < c2)
        seq_array = np.array([[dictt[x] for x in seq],]*400) #(400,Length)
        aa = 0
        names = np.array([seq,]*400) #(400)
        for i in range(0,400,20):
            for j in range(0,20):
                seq_array[i+j,c1] = aa #c1 resi
                seq_array[i+j,c2] = j #c2 resi
                names[i+j] = names[i+j][:c1]  + dictt_inv[aa] + names[i+j][c1+1:]
                names[i+j] = names[i+j][:c2]  + dictt_inv[j] + names[i+j][c2+1:]
            aa += 1
        seq_tensor = np.transpose(np.eye(20)[seq_array.T],(1,2,0)) # (400, 20, Length)
        seq_hydropath = np.matmul(np.array([hydropath.T,]*400) , seq_tensor) #(400, 4, Length)
        combined = np.mean(seq_hydropath,2)
        names = np.reshape(names,(400,1))
        return seq_array,seq_hydropath,combined,names,[0,]*400

def make_batch8000(seq,resnum,per=0):
        num = 20**len(resnum) #resnum example is 3
        seq_array = np.array([[dictt[x] for x in seq],]*num) #(8000,Length)
        names = np.array([seq,]*num) # ignored
        seq_array[:,resnum] = np.array(recursive_tree(len(resnum)))
        seq_tensor = np.transpose(np.eye(20)[seq_array.T],(1,2,0)) # (400, 20, Length)
        seq_hydropath = np.matmul(np.array([hydropath.T,]*num) , seq_tensor) #(400, 4, Length)
        combined = np.mean(seq_hydropath,2)
        names = np.reshape(names,(num,1))
        return seq_array,seq_hydropath,combined,names,[0,]*num

def recursive_tree(seq,ans = []):
     if len(ans)  == seq -1:
         return [ans+[x,] for x in range(0,20)]
     else:
        temp = []
        for i in range(0,20):
            temp += recursive_tree(seq,ans + [i,])
        return  temp

p53_seq='ETFSDLWKLLPEN'
unchanged=(2,6,9)
iterate  = {}
for i in range(0,13):
    iterate[i] = []
    if i not in (2,6,9): # exclude impt residues
        for j in range(0,20):
            iterate[i] += [[i,j],]
resnum ,aa = [],[]
result = []
def string(arr):
    result = ''
    for i in arr:
        result += dictt_inv[i]
    return result
        
def test(seq):
    p53 = make(seq)
    X = [(p53[0],p53[1],p53[2])]
    y = [p53[-1],]
    Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(X,y,0)
    b = sess.run (out_softmax, feed_dict={Inp0: Inp0_,Inp1: Inp1_,Inp2: Inp2_,dropout : 1,learning_rate : lr})
    return b,p53

import os
all_models = [x[:-5] for x in os.listdir('.') if ('.ckpt.meta' in x and x.startswith('model2'))]
dictt_model = {}
for i1 in range(0,4): #test set
    for i2 in range(0,4): #CV set
        for j in all_models: #find if model they belong to that set
            if j[7] == str(i1) and  j[9] == str(i2) :
                if (i1,i2) not in dictt_model:
                    dictt_model[(i1,i2)] = []
                else:
                    dictt_model[(i1,i2)] += [j,]
for c1 in range(7,13): #first three res #start from4
    for c2 in range(c1+1,13):
        for c3 in range(c2+1,13):
             if c2 not in unchanged and c1 not in unchanged and c3 not in unchanged: #readable but slower
                print 'Run :',c1,c2,c3,
                Inp0_,Inp1_,Inp2_,names,labels_ = make_batch8000(p53_seq,(c1,c2,c3))
                data= pd.DataFrame(Inp0_,columns=range(13))
                data['prob']  = 0
                counter =0 
                for test_fold in dictt_model:
                    for file in  sorted(dictt_model[test_fold],
                        key = lambda x : np.float(x[:-5].split('_')[1:][-2]))[-3:]:
                        saver.restore(sess,file)
                        b = sess.run (out_softmax, feed_dict={Inp0: Inp0_,
                                                                        Inp1: Inp1_,Inp2: Inp2_,
                                                                        dropout : 1,learning_rate : lr})
                        thres = 0.7
                        data['prob'] = np.reshape(b,(8000)) + data['prob']
                        data['fold'+str(counter) ] = b
                        counter += 1
                data['prob'] =  data['prob'] /counter
                data.to_csv('results/results_%s_%s_%s.csv' %(c1,c2,c3),index=0)
                print data.sort_values('prob')[-5:]['prob'].values
##                for mod_seq,prob in zip(pd.DataFrame(Inp0_,columns=range(13))[b>thres].values,b[b>thres]):
##                    print p53_seq
##                    mod_seq = string(mod_seq)
##                    print mod_seq,prob,'\n'
                    #print test(mod_seq)[0]

np.save('p53_3_mutations.npy',result)            
die

p53 = make(p53_seq)
XX = [(p53[0],p53[1],p53[2]),]
yy = [p53[-1],]
Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(XX,yy,0)
a,b, = sess.run ([acc, out_softmax], feed_dict={Inp0: Inp0_,
                                                                Inp1: Inp1_,Inp2: Inp2_,
                                                                labels: labels_,
                                                                dropout : 1,learning_rate : lr})
energy = b[0][0]
Temp = 0.2
result = []
die
for i in range(0,100000):
    p53_seq_new= mutate(p53_seq)
    p53 = make(p53_seq)
    XX = [(p53[0],p53[1],p53[2]),]
    yy = [p53[-1],]
    Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(XX,yy,0)
    a,b, = sess.run ([acc, out_softmax], feed_dict={Inp0: Inp0_,
                                                                    Inp1: Inp1_,Inp2: Inp2_,
                                                                    labels: labels_,
                                                                    dropout : 1,learning_rate : lr})

    proposed_E = b[0][0]
    if np.random.uniform() <= np.exp((proposed_E-energy)/Temp):
        p53_seq = p53_seq_new
        energy = proposed_E
        if  energy >= 0.9:
            result += [p53_seq,energy,]
            print p53_seq, energy


die

