
# coding: utf-8

# In[1]:

#%matplotlib inline
import matplotlib.pyplot as plt

import keras
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D ,AveragePooling2D

import tensorflow as tf

import os
import pickle
import numpy as np


data = np.load('x_data_100_classes_5k.npy.zip')['x_data_100_classes_5k']
data = np.array(map(lambda x : np.reshape(x,(28,28,1)) , data))

classes = ['airplane','alarm clock','ambulance','angel','ant','anvil','apple','axe','banana','bandage','barn','baseball bat','baseball',
           'basket','basketball','bathtub','beach','bear','beard','bed','bee','belt','bicycle','binoculars','birthday cake','blueberry',
           'book','boomerang','bottlecap','bowtie','bracelet','brain','bread','broom','bulldozer','bus','bus','butterfly','cactus','cake',
           'calculator','calendar','camel','camera','campfire','candle','cannon','canoe','car','carrot','cello','computer',
           'cat','chandelier','clock','cloud','coffee cup','compass','cookie','couch','cow','crab','crayon','crocodile','crown',
           'cup','diamond','dog','dolphin','donut','dragon','dresser','drill','drums','duck','dumbbell','ear','elbow',
           'elephant','envelope','eraser','eye','eyeglasses','face','fan','feather','fence','finger','fire hydrant',
           'fireplace','firetruck','fish','flamingo','flashlight','flip flops','floor lamp','flower','flying saucer',
           'foot','fork']

#plt.imshow(data[9999]);plt.show()
#plt.imshow(data[10000]);plt.show()
y = np.zeros(500000 ,dtype = np.uint8)
label = 0
counter = 0
for i in range(len(data)):
    if classes[label] == 'Bus':
        y[i] = 35
    else:
        y[i] = label
    counter += 1
    if counter==5000:
        counter = 0
        label += 1
    
batch_size = 320
num_classes = 100
epochs = 50

x_train ,y_train = data[0::2],y[0::2]
x_test, y_test = data[1::2],y[1::2]
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


Inp0 = Input(shape=(28,28,1),name = 'Input_layer')
Inp = keras.layers.BatchNormalization()(Inp0)
#ConvBlock 01

conv01a = Conv2D(32, (3, 3), padding='same',activation = 'relu', input_shape=Inp.shape,name = 'Conv01_layerA')(Inp)
conv02a = Conv2D(32, (3, 3),activation = 'relu',name = 'Conv02_layerA')(conv01a)
maxpool_01a = MaxPooling2D(pool_size=(2, 2),name = 'MaxPool01_layerA')(conv02a)
drop01a = Dropout(0.25,name = 'Dropout01_layerA')(maxpool_01a)

conv01b = Conv2D(32, (3, 3), padding='same',activation = 'relu', input_shape=Inp.shape,name = 'Conv01_layerB')(Inp)
conv02b = Conv2D(32, (3, 3),activation = 'relu',name = 'Conv02_layerB')(conv01b)
avgpool_01b = AveragePooling2D(pool_size=(2, 2),name = 'AvgPool01_layerB')(conv02b)
drop01b = Dropout(0.25,name = 'Dropout01_layerB')(avgpool_01b)
drop01_p = keras.layers.concatenate([drop01a,drop01b])

#Convblock 02
drop01 = keras.layers.BatchNormalization()(drop01_p)
conv03a = Conv2D(64, (3, 3), padding='same',activation = 'relu',name = 'Conv03_layerA')(drop01)
conv04a = Conv2D(64, (3, 3),activation = 'relu',name = 'Conv04_layerA')(conv03a)
maxpool_02a = MaxPooling2D(pool_size=(2, 2),name = 'MaxPool02_layerA')(conv04a)
drop02a = Dropout(0.25,name = 'Dropout02_layerA')(maxpool_02a)

conv03b = Conv2D(64, (3, 3), padding='same',activation = 'relu',name = 'Conv03_layerB')(drop01)
conv04b = Conv2D(64, (3, 3),activation = 'relu',name = 'Conv04_layerB')(conv03b)
Avgpool_02b = AveragePooling2D(pool_size=(2, 2),name = 'AvgPool02_layerB')(conv04b)
drop02b = Dropout(0.25,name = 'Dropout02_layerB')(Avgpool_02b)

drop02_p = keras.layers.concatenate([drop02a,drop02b])
#ConvBLock 03
drop02 = keras.layers.BatchNormalization()(drop02_p)
conv05 = Conv2D(256, (3, 3),activation = 'relu',name = 'Conv05_layer')(drop02)
conv06 = Conv2D(256, (2, 2),activation = 'relu',name = 'Conv06_layer')(conv05)
drop03 = Dropout(0.25,name = 'Dropout03_layer')(conv06)

# Fully Connected Dense block
x = Flatten(name = 'Flatten_layer')(drop03)
x = Dense(512, name = 'Dense_layer')(x)
x = Activation('relu',name='Dense_Relu') (x)
x = Dropout(0.5,name = 'Dropout04_layer')(x)
logits_layer = Dense(num_classes, name= 'logits_layer')(x)
output = Activation('softmax',name = 'Sofftmax_layer')(logits_layer)

# Define model inputs and output
model = Model(Inp0, output)
model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.00003, decay=1e-4) #decays by two orders of magnitude

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[2]:

#f, ax = plt.subplots(20,20,figsize=(40,40))
counter= 0
for i in range(356,500000,5000//4):
    #ax[counter//20,counter%20].imshow(data[i,:,:,0])
    counter += 1
#plt.imshow(data[10000]);plt.show()
#plt.show()


# In[3]:

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[4]:

model.load_weights('./weight2.h5')


# In[5]:

##hist = model.fit(x_train[0::100], y_train[0::100],batch_size,
##                           1,verbose = 1,
##                           validation_data=(x_test[0::100], y_test[0::100]))


# In[6]:

pred = model.predict(x_test[::100],1000)
X_pred = x_test[::100]
y_pred = y_test[::100]

# In[7]:

pred2 = map( lambda x: x.argsort()[-5:],pred)


# In[8]:

classes = ['airplane','alarm clock','ambulance','angel','ant','anvil','apple','axe','banana','bandage','barn','baseball bat','baseball',
           'basket','basketball','bathtub','beach','bear','beard','bed','bee','belt','bicycle','binoculars','birthday cake','blueberry',
           'book','boomerang','bottlecap','bowtie','bracelet','brain','bread','broom','bulldozer','bus','bus','butterfly','cactus','cake',
           'calculator','calendar','camel','camera','campfire','candle','cannon','canoe','car','carrot','cello','computer',
           'cat','chandelier','clock','cloud','coffee cup','compass','cookie','couch','cow','crab','crayon','crocodile','crown',
           'cup','diamond','dog','dolphin','donut','dragon','dresser','drill','drums','duck','dumbbell','ear','elbow',
           'elephant','envelope','eraser','eye','eyeglasses','face','fan','feather','fence','finger','fire hydrant',
           'fireplace','firetruck','fish','flamingo','flashlight','flip flops','floor lamp','flower','flying saucer',
           'foot','fork']


# # THE INCORRECT ONES

# In[9]:

### plot those that are incorrect
##for i in range(19,2500,20):
##    if np.argmax(y_pred[i]) not in pred2[i] :
##        plt.imshow(X_pred[i,:,:,0]);
##        print sorted(pred[i])[-5:],map( lambda x : classes[x],pred2[i]),classes[np.argmax(y_pred[i])]
##        plt.show();plt.close()
##
### # THE CORRECT ONES
##
### In[10]:
##
### plot those that are correct
##for i in range(19,2500,20):
##    if np.argmax(y_pred[i])  in pred2[i] :
##        plt.imshow(X_pred[i,:,:,0]);
##        print sorted(pred[i])[-5:],map( lambda x : classes[x],pred2[i]),classes[np.argmax(y_pred[i])]
##        plt.show();plt.close()



# In[11]:

from keras import backend as K #clear memory
#K.clear_session()


# # INCEOPTION3 

# In[12]:

from keras.applications.inception_v3 import *


# In[13]:

base_model = InceptionV3(weights='imagenet', include_top=False)


# In[24]:

base_model.summary()


# In[43]:

mixed4 = base_model.get_layer('mixed2').output
mixed_model = Model(base_model.input,mixed4)


# In[44]:

mixed4


# In[ ]:
array1 = np.zeros(shape=(250000,288)) #embeddings for train
array2 = np.zeros(shape=(250000,288)) #embeddings for test
jump = 50000
for i in range(0,250000,jump):
    print i
    temp = x_train[i:i+jump]
    temp = np.concatenate([temp,temp,temp],-1)
    array1[i:i+jump] = np.reshape(mixed_model.predict(temp[:jump],7500),(jump,288))

    temp = x_test[i:i+jump]
    temp = np.concatenate([temp,temp,temp],-1)
    array2[i:i+jump] = np.reshape(mixed_model.predict(temp[:jump],7500),(jump,288))
#X_train2 = np.concatenate([x_train[::10],x_train[::10],x_train[::10]],-1)

# lets try for the incorrect ones
for i in range(19,2500,20):
    if np.argmax(y_pred[i]) not in pred2[i] :
        f, ax = plt.subplots(1,4)
        temp = X_pred[i:i+1];temp = np.concatenate([temp,temp,temp],-1)
        vec = np.reshape(mixed_model.predict(temp[:jump]),(1,288))
        dist = map(lambda x : np.mean((x - vec)**2), array2[:])
        closest_pic = np.argsort(dist)[:3]
        ax[0].imshow(X_pred[i,:,:,0])
        counter = 1
        for j in closest_pic :
            ax[counter].imshow(x_test[j,:,:,0])
            ax[counter].set_xlabel('label:%s\ndist:%s'%(classes[np.argmax(y_test[j])],dist[j]))
            counter += 1
            
        plt.show()
        print sorted(pred[i])[-5:],map( lambda x : classes[x],pred2[i]),classes[np.argmax(y_pred[i])]
        plt.close()

# In[45]:

mixed_model.predict(X_train2[0:1])


# In[24]:



