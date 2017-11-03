# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import  numpy as np
import tensorflow as tf
import pandas as pd
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
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
x=tf.Variable(np.array([[hydropath,hydropath*5,hydropath*10],[hydropath*35,hydropath*555,hydropath*5555]]),dtype=tf.float32)
x = tf.Variable(hydropath,dtype=tf.float32)
mean,var = tf.nn.moments(x,[0,1],keep_dims=True)
scale = tf.Variable(tf.ones([1]))
beta = tf.Variable(tf.zeros([1]))
epsilon = 1e-3
y = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon,name='name')
init = tf.global_variables_initializer();sess = tf.Session();sess.run(init)
