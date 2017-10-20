# -*- coding: utf-8 -*-
import  numpy as np
import tensorflow as tf
f1= open('SVM_data','r')
result = []
for line in f1:
    if '[' not in line:
        #print line,
        result += [line[:-1],]

dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}
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
'D' : [ 	1.23 ,	3.64, 	2.41 ,-1]}
hydropath_embed =\
np.array([[0.170, 0.500, 0.330, 0.000],
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

'''
SQETFSDLWKLLPEN
'''
X = []
counter = 0
for i in result:
    temp = np.zeros((len(i),5))
    for j in  range(len(i)):
        res = dictt[i[j]]
        temp[j][0] = res
        #temp[j][-1] = len(i)*0.01
        temp[j][-4:] = dictt_hydropathy[i[j]]
    alternative = [len(i),np.sum(temp[-1]),np.sum(temp[-2]),np.sum(temp[-3])] 
    if counter <= 110:
        per = 1
    else:
        per = 0
    temp = temp.T

    X += [[temp[0],temp[1:],alternative,i,per],]
    #print '> %s\n%s' %(i,i)
    counter += 1
        

print len(result)

epsilon = 1e-3
def batch_normalization(x,name='batchnorm',feature_norm = False):
    #epsilon = tf.Variable(tf.constant([1e-3,]*x.shape[0]))
    if feature_norm : #only works for fizxed batch size 1
        mean,var = tf.nn.moments(x,[0,2,3],keep_dims=True)
        scale = tf.Variable(tf.ones([1,x.shape[1],1,1]))
        beta = tf.Variable(tf.zeros([1,x.shape[1],1,1]))
        x = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon,name=name)
    else:
        mean,var = tf.nn.moments(x,[0,1,2],keep_dims=False)
        scale = tf.Variable(tf.ones([x.shape[-1]]))
        beta = tf.Variable(tf.zeros([x.shape[-1]]))
        x = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon,name=name)
    return x

learning_rate = 0.00001
training_epochs = 100
batch_size = 1
display_step = 1
n_classes =5
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
with tf.name_scope('inputs') as scope:
    Inp0 = tf.placeholder(tf.int32,[None,None],name='sequence_factors1')
    #Inp0b = tf.Variable(Inp0,tf.float32)
    Inp0b = tf.Variable([[14, 12, 2, 16, 5, 14, 3, 10, 17, 8, 10, 10, 13, 2, 11]],dtype=tf.int32,name='sequence_factors11')
    #Inp1 = tf.placeholder(tf.float32,[None,4,None],name='sequence_factors2')
    labels = tf.placeholder(tf.float32 , [None,1],name='labels')
    dropout = tf.placeholder(tf.float32,name='dropout')
    #Inp2 = tf.placeholder(tf.float32, [None,4] ,name = 'globa_seq_info') # converted to embeddings
    hydro_embeddings = tf.Variable(hydropath_embed ,name = 'aa_embeddings', trainable = False, dtype = tf.float32)
    embedded_word_ids = tf.gather(hydro_embeddings,range(0,20))
    Inp2_a = tf.nn.embedding_lookup(hydro_embeddings,Inp0b,name='lookup')
    Inp2 = tf.reduce_mean(Inp2_a,axis=1,name = 'globa_seq_info')
    Inp1 = tf.transpose(Inp2_a, (0,2,1), name= 'seq_factor2_embeddings')
with tf.name_scope('embedding') as scope:
    aa_embeddings = tf.get_variable('aa_embeddings',[20, 5])
    embedded_word_ids = tf.gather(aa_embeddings,range(0,20))
    embed0 = tf.nn.embedding_lookup(aa_embeddings,Inp0b,name='lookup')
    embed1 = tf.transpose(embed0,(0,2,1))
    unstack0 = tf.unstack(Inp1,axis=-2,name='unstack0')
    unstack1 = tf.unstack(embed1 , axis=-2,name='unstack1')
    layer0 = tf.stack([tf.stack(unstack0+unstack1,axis=1)],-1,name='stack')

with tf.name_scope('layer1') as scope:
    layer1_norm = batch_normalization(layer0,'BN_layer0',feature_norm = True)
    layer1 = tf.layers.conv2d(layer1_norm,128,(9,1),padding='valid',activation=tf.nn.relu,trainable = False)
    layer1_DO = tf.layers.dropout(layer1,rate=dropout,name='Drop1')

with tf.name_scope('layer2') as scope:
    layer2_norm = batch_normalization(layer1_DO,'BN_layer1')
    layer2 = tf.layers.conv2d(layer2_norm,128,(1,7),padding='same',activation=tf.nn.relu)
    layer2_DO = tf.layers.dropout(layer2,rate=dropout,name='Drop2')

with tf.name_scope('layer3') as scope:
    layer3_norm = batch_normalization(layer2_DO,'BN_layer1')
    layer3 = tf.layers.conv2d(layer3_norm,128,(1,7),padding='same',activation=tf.nn.relu)
    layer3_DO = tf.layers.dropout(layer3,rate=dropout,name='Drop3')

with tf.name_scope('layer4') as scope:
    layer4_norm = batch_normalization(layer3_DO,'BN_layer2')
    layer4 = tf.layers.conv2d(layer4_norm,128,(1,3),padding='same',activation=tf.nn.relu)
    layer4_DO = tf.layers.dropout(layer4,rate=dropout,name='Drop4')

with tf.name_scope('dense') as scope:
    globalmeanpooling = tf.reduce_mean(layer4[:,:,:,0::2],(1,2),name='globalmeanpooling')
    globalmaxpooling = tf.reduce_mean(layer4[:,:,:,1::2],(1,2),name='globalmaxpooling')
    gbmp_extra = tf.concat([Inp2,globalmaxpooling,globalmeanpooling],axis = 1, name ='gbmp_extra')
    layer5_DO = tf.layers.dropout(gbmp_extra,rate=dropout,name='Drop5')
    dense1 = tf.layers.dense(layer5_DO,4 , name = 'dense1' )
    layer6_DO = tf.layers.dropout(dense1,rate=dropout,name='Drop6')
    dense2 = tf.layers.dense(layer6_DO,1 , name = 'dense2' )
##    dense3 = tf.layers.dense(dense2,1 , name = 'dense3' )
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
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.5).minimize(mean_loss,var_list=tf.trainable_variables())
with tf.name_scope('accuracy') as scope:
    predict_boo = tf.greater(out_softmax,0.5)
    predict = tf.cast(predict_boo, np.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels,predict),tf.float32),name='accuracy')
var_grad = tf.gradients(loss, tf.trainable_variables()[0:1])
import sklearn.metrics,random
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)


saver.restore(sess,'model.ckpt')


RESULT = {}
import sys
if len(sys.argv)>=3:
    fold, test = int(sys.argv[1]), int(sys.argv[2])
else:
    fold, test = 0,2
assert fold != test
all_data =[]
test_emsemble= []
for CV in range(10):
  if CV != test:
    init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
    RESULT[CV] = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []    
    for i in range(len(X)):
        x = X[i]
        if i%5 == CV:
            X_val += [(x[0],x[1],x[2]),]
            y_val += [x[-1],]
            if x[-1] == 0:
                X_val += [(x[0],x[1],x[2]),(x[0],x[1],x[2]),]
                y_val += [x[-1],x[-1],]
        elif i%5 == test:
            X_test += [(x[0],x[1],x[2]),]
            y_test += [x[-1],]
            if x[-1] == 0:
                X_test += [(x[0],x[1],x[2]),(x[0],x[1],x[2]),]
                y_test += [x[-1],x[-1],]
        else:
            X_train += [(x[0],x[1],x[2]),]
            y_train += [x[-1],]
            if x[-1] == 0:
                X_train += [(x[0],x[1],x[2]),(x[0],x[1],x[2]),]
                y_train += [x[-1],x[-1],]
    best_roc_val = {}       
    for epoch in range(training_epochs):#training_epochs):
        if epoch%20 ==0 :
            init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
        logit_train = []
        cost_train = []
        random.seed(epoch)
        shuffle = range(len(X_train))
        random.shuffle(shuffle)
        counter = 0
        for i in shuffle:
            lr = ((1+np.cos(1.0*counter*3.142/len(shuffle)))**3)*0.0007*((51.0-epoch)/50)**2
            counter += 1
            #print i
            Inp0_ = np.array([X_train[i][0]])
            Inp1_ = np.array([X_train[i][1]])
            labels_ = np.array([[y_train[i],]])
            _, c = sess.run([optimizer, acc], feed_dict={Inp0: Inp0_,
                                                               #Inp1: Inp1_,
                                                               labels: labels_,
                                                               dropout : 0.4,learning_rate : lr})
            die

        logit_train = []
        cost_train = []
        lr = 0
        for i in range(len(X_train)):
            Inp0_ = np.array([X_train[i][0]])
            Inp1_ = np.array([X_train[i][1]])
            labels_ = np.array([[y_train[i],]])
            c = sess.run (acc, feed_dict={Inp0: Inp0_,
                                          Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})
            out = sess.run( out_softmax, feed_dict={Inp0: Inp0_,
                                                  Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})[0][0]
            cost_train += [c,]
            logit_train += [ out,]
        logit_val = []
        cost_val = []
        for i in range(len(X_val)):
            Inp0_ = np.array([X_val[i][0]])
            Inp1_ = np.array([X_val[i][1]])
            labels_ = np.array([[y_val[i],]])
            c = sess.run (acc, feed_dict={Inp0: Inp0_,
                                                  Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})
            out = sess.run( out_softmax, feed_dict={Inp0: Inp0_,
                                                  Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})[0][0]
            cost_val += [c,]
            logit_val += [ out,]
        logit_test = []
        cost_test = []
        for i in range(len(X_test)):
            Inp0_ = np.array([X_test[i][0]])
            Inp1_ = np.array([X_test[i][1]])
            labels_ = np.array([[y_test[i],]])
            c = sess.run (acc, feed_dict={Inp0: Inp0_,
                                                  Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})
            out = sess.run( out_softmax, feed_dict={Inp0: Inp0_,
                                                  Inp1: Inp1_,
                                                  labels: labels_,
                                                  dropout : 1,learning_rate : lr})[0][0]
            cost_test += [c,]
            logit_test += [ out,]
        roc_train = sklearn.metrics.roc_auc_score(y_train,logit_train)
        roc_val   = sklearn.metrics.roc_auc_score(y_val,logit_val)
        roc_test   = sklearn.metrics.roc_auc_score(y_test,logit_test)
        #print np.mean(cost_train),np.mean(cost_val)
        print roc_train,roc_val,roc_test
        best_roc_val[epoch] = [roc_train,roc_val,roc_test,logit_test]
        all_data += [[roc_train,roc_val,roc_test],]
        #best_logit_test += [logit_test,]
##            save_path = saver.save(sess,'model_%s_%s_%s_%s_%s_%s.ckpt' \
##                                   %(fold,test,epoch,int(100*np.mean(best_roc_val[0])),
##                                     int(100*np.mean(best_roc_val[1])),int(100*np.mean(best_roc_val[2]))))
        if roc_val <= 0.5:
            None#init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
        RESULT[CV] += [[roc_train,roc_val,roc_test],]
        #print logit_val
    best_logit_test = sorted([best_roc_val[ep] for ep in best_roc_val], key = lambda x :x[1])[-3:]
    for j in best_logit_test:
        test_emsemble += [j[3],]
    print sklearn.metrics.roc_auc_score(y_test,np.mean(np.array(test_emsemble),0))
    save_path = saver.save(sess,'model.ckpt')
        
        
        
        
        
