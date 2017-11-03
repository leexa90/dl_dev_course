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

for idx in range(len(data)):
    i = data.iloc[idx]['seq']
    temp = np.zeros((len(i),5))
    for j in  range(len(i)):
        res = dictt[i[j]]
        temp[j][0] = res
        #temp[j][-1] = len(i)*0.01
        temp[j][-4:] = dictt_hydropathy[i[j]]
    alternative = [len(i),np.sum(temp[-1]),np.sum(temp[-2]),np.sum(temp[-3])]
    per = data.iloc[idx]['source'] == 2 
    temp = temp.T
    X += [[temp[0],temp[1:],alternative,i,per*1],]
    #print '> %s\n%s' %(i,i)
    counter += 1
data['X'] = X

batch_size = 1
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

with tf.name_scope('inputs') as scope:
    Inp0 = tf.placeholder(tf.int32,[batch_size,None],name='sequence_factors1')
    Inp1 = tf.placeholder(tf.float32,[batch_size,4,None],name='sequence_factors2')
    labels = tf.placeholder(tf.float32 , [batch_size,1],name='labels')
    dropout = tf.placeholder(tf.float32,name='dropout')
    Inp2 = tf.placeholder(tf.float32, [batch_size,4] ,name = 'globa_seq_info')

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
    layer1 = tf.layers.conv2d(layer0,32,(9,1),padding='valid',activation=tf.nn.relu)
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
print len(result)
training_epochs  = 100

import sklearn.metrics,random


RESULT = {}
import sys
if len(sys.argv)==2:
    test = int(sys.argv[1])
else:
    test = 2
print "TEST IS",test
all_data =[]
test_emsemble= []
def get_data_from_X(X,y,i): #get tensor inputs from X and y
    Inp0_ = np.array([X[i][0]])
    Inp1_ = np.array([X[i][1]])
    Inp2_ = np.array([X[i][2]])
    labels_ = np.array([[y[i],]])
    return Inp0_,Inp1_,Inp2_,labels_
folds= 5
# ensure no leakage of train to test, was getting 95 AUC
X_test = []
y_test = []
print 'train_val_test size:' , len(X)
for i in range(test,len(X),folds):
    if i%5 == test:
        x = X[i]
        X_test += [(x[0],x[1],x[2]),]
        y_test += [x[-1],]
        X[i] = 'test'
X = [x for x in X if x != 'test']
print 'train+val  size :', len(X)
saver = tf.train.Saver( max_to_keep=5000)
repeat = 0
for repeat in range(0,1): #perform 5 repeats
    for CV in range(folds-1): #for each repeat, do 4 fold CV. (test set is kept constant throughtout)# 
        init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
        RESULT[CV] = []
        X_train = []
        y_train = []
        X_val = []
        y_val = []    
        for i in range(len(X)):
            x = X[i]
            if i%5 == CV:
                X_val += [(x[0],x[1],x[2]),]
                y_val += [x[-1],]
            else:
                X_train += [(x[0],x[1],x[2]),]
                y_train += [x[-1],]

        best_roc_val = {}     # stores val ROC for trainnig epochs per CV+repeat run
        for epoch in range(training_epochs):#training_epochs):
            if epoch%20 ==0 :
                init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
            logit_train = []
            cost_train = []
            random.seed(epoch)
            shuffle = range(len(X_train)) #shuffle index of X_train
            random.shuffle(shuffle)
            counter = 0
            for i in shuffle[::10]: #training with bagging
                lr = ((1+np.cos(1.0*counter*3.142/len(shuffle)))**3)*0.0007*((51.0-epoch)/50)**2
                counter += 1
                Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(X_train,y_train,i)
                if labels_[0][0] == 1:
                    lr = lr * 8.5 #reweigh LR for pos
                else:
                    lr = lr * 0.5 #reweigh LR for neg
                
                _, c = sess.run([optimizer, acc], feed_dict={Inp0: Inp0_,Inp2: Inp2_,
                                                                   Inp1: Inp1_,
                                                                   labels: labels_,
                                                                   dropout : 0.4,learning_rate : lr}) #sgd 


            logit_train = []
            cost_train = []
            lr = 0 
            for i in range(len(X_train)): #train error
                Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(X_train,y_train,i)
                c, out = sess.run ([acc, out_softmax], feed_dict={Inp0: Inp0_,
                                              Inp1: Inp1_,Inp2: Inp2_,
                                                      labels: labels_,
                                                      dropout : 1,learning_rate : lr})
                cost_train += [c,]
                logit_train += [ out[0][0],]
            logit_val = []
            cost_val = []
            for i in range(len(X_val)): #val error
                Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(X_val,y_val,i)
                c, out = sess.run ([acc, out_softmax], feed_dict={Inp0: Inp0_,
                                              Inp1: Inp1_,Inp2: Inp2_,
                                                      labels: labels_,
                                                      dropout : 1,learning_rate : lr})
                cost_val += [c,]
                logit_val += [ out[0][0],]
            logit_test = []
            cost_test = []
            for i in range(len(X_test)): #test errror
                Inp0_,Inp1_,Inp2_,labels_ = get_data_from_X(X_test,y_test,i)
                c, out = sess.run ([acc, out_softmax], feed_dict={Inp0: Inp0_,
                                              Inp1: Inp1_,Inp2: Inp2_,
                                                      labels: labels_,
                                                      dropout : 1,learning_rate : lr})
                cost_test += [c,]
                logit_test += [ out[0][0],]
            roc_train = sklearn.metrics.roc_auc_score(y_train,logit_train)
            roc_val   = sklearn.metrics.roc_auc_score(y_val,logit_val)
            roc_test   = sklearn.metrics.roc_auc_score(y_test,logit_test)
            print np.mean(cost_train),np.mean(cost_val)
            print roc_train,roc_val,roc_test
            best_roc_val[epoch] = [roc_train,roc_val,roc_test,logit_test]
            all_data += [[roc_train,roc_val,roc_test],]
            #best_logit_test += [logit_test,]
    ##            save_path = saver.save(sess,'model_%s_%s_%s_%s_%s_%s.ckpt' \
    ##                                   %(fold,test,epoch,int(100*np.mean(best_roc_val[0])),
    ##                                     int(100*np.mean(best_roc_val[1])),int(100*np.mean(best_roc_val[2]))))
            if roc_val <= 0.5: # if val worst then chance, reinitalize tarining
                init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
            best_logit_test = sorted([best_roc_val[ep] for ep in best_roc_val], key = lambda x :x[1])[-3:]
            if len(best_logit_test) >=3 and best_logit_test[0][1] <= roc_val:
                model_name = 'model2_%s_%s_%s_%s_%s_%s.ckpt' %(test,CV,repeat,str(roc_train)[:5],str(roc_val)[:5],str(roc_test)[:5])
                saver.save(sess,model_name),
                print 'SAVED\n'
        for j in best_logit_test:
            test_emsemble += [j[3],]
        print [x[1:3] for x in best_logit_test]
        print sklearn.metrics.roc_auc_score(y_test,np.mean(np.array(test_emsemble),0))

	
        
        
        
