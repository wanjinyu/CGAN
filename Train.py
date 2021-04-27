# -*- coding: utf-8 -*-
"""
Created on Mon Nov  10 14:55:27 2020

@author: Wanjinyu
"""
import tensorflow as tf
import numpy as np
import CGAN
import h5py
import scipy.io as sio

print(tf.test.is_gpu_available())
X_dim = 3
y_dim = 200
Z_dim = 100
G_dim = 500
G_dim = 500
mb_size = 64
Ndata = 10000
lr_G = 0.0001
lr_D = 0.0001
#load data 
# X: data samples, Y: labels
filename='Y.mat'
data_all = sio.loadmat(filename)
density = data_all['Y']

filename='X.mat'
data_all = sio.loadmat(filename)
x = data_all['X']
Nsample = len(x)
# nomarlization
x_range = np.zeros([X_dim,2])
for i in range(X_dim):
    x_range[i,0]=x[:,i].min()
    x_range[i,1]=x[:,i].max()

y_range = np.zeros([y_dim,2])
for i in range(y_dim):
    y_range[i,0]=density[:,i].min()
    y_range[i,1]=density[:,i].max()

Xdata = np.zeros([Nsample,X_dim])
Ydata = np.zeros([Nsample,y_dim])
for i in range(X_dim):
    Xdata[:,i] = (x[:,i]-x_range[i,0])/(x_range[i,1]-x_range[i,0])
for i in range(y_dim):
    Ydata[:,i] = (density[:,i]-y_range[i,0])/(y_range[i,1]-y_range[i,0])
    
Ydata = Ydata.reshape([Nsample,y_dim])
Xdata = Xdata.reshape([Nsample,X_dim])

idx = np.arange(Nsample)
np.random.shuffle(idx)

Xtrain = Xdata[idx[0:Ndata],:]
Ytrain = Ydata[idx[0:Ndata],:]


CG_nets = CGAN.CGAN(X_dim, y_dim, Z_dim, G_dim, G_dim)

X = tf.placeholder(tf.float32,shape=[None,X_dim])
y = tf.placeholder(tf.float32,shape=[None,y_dim])
Z = tf.placeholder(tf.float32, shape=[None,Z_dim])

G_sample = CG_nets.generator(Z,y)
D_real, D_logit_real = CG_nets.discriminator(X,y)
D_fake, D_logit_fake = CG_nets.discriminator(G_sample,y)
# loss function
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits\
                              (logits=D_logit_real, labels=tf.ones_like(D_logit_real))) 
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits\
                              (logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake))) 
D_loss = D_loss_real+D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits\
                        (logits=D_logit_fake,labels=tf.ones_like(D_logit_fake)))
D_solver = tf.train.AdamOptimizer(learning_rate=lr_D).minimize(D_loss)
G_solver = tf.train.AdamOptimizer(learning_rate=lr_G).minimize(G_loss)

# batch-to-batch  
dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
dataset = dataset.shuffle(1000).batch(mb_size).repeat()
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# training
Config=tf.ConfigProto(allow_soft_placement=True)  
Config.gpu_options.allow_growth=True
sess = tf.Session(config=Config)
sess.run(tf.global_variables_initializer())
i=0
G_history = []
D_history = []
Nepochs = 10000
for it in range(1000000):
    X_mb, y_mb =  sess.run(next_element)
    Z_sample = CG_nets.sample_Z(len(X_mb[:,0]), Z_dim)
    # train D an G 
    _, D_loss_curr = sess.run([D_solver,D_loss],feed_dict={X:X_mb, Z:Z_sample, y:y_mb})   
    _, G_loss_curr = sess.run([G_solver,G_loss],feed_dict={Z:Z_sample, y:y_mb}) 
    D_history.append(D_loss_curr)
    G_history.append(G_loss_curr)
    if it%100 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()