#%matplotlib inline
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.load_weights('./weight.h5')
from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_acc', patience=5, verbose=0),
#    ModelCheckpoint(filepath='./weights.h5',
#                                    monitor='val_acc',
#                                    verbose=1, save_best_only=True)
    ]

hist = model.fit(x_train, y_train,batch_size=batch_size,
                           epochs=epochs,verbose = 2,
                           validation_data=(x_test, y_test),callbacks=callbacks)
model.save_weights('./weight2.h5')
np.save('hist2.npy',hist.history)
##hist = model.fit_generator(datagen.flow(x_train, y_train,
##                                 batch_size=batch_size),
##                    steps_per_epoch=x_train.shape[0] // batch_size,
##                    epochs=epochs,
##                    validation_data=(x_test, y_test),
##                    workers=4)
