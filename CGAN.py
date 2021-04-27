# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:15:01 2020

@author: Wanjinyu
"""
import tensorflow as tf
import numpy as np

class CGAN:
    def __init__(self, X_dim, y_dim, Z_dim, G_dim, D_dim):
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.Z_dim = Z_dim
        self.G_dim = G_dim
        self.D_dim = D_dim

        self.D_W1 = tf.Variable(self.xavier_init([X_dim+y_dim,D_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[D_dim])) 
        self.D_W2 = tf.Variable(self.xavier_init([D_dim,1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.G_W1 = tf.Variable(self.xavier_init([Z_dim+y_dim,G_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[G_dim]))
        self.G_W2 = tf.Variable(self.xavier_init([G_dim,X_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
        
    # randomness
    def xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1./tf.sqrt(in_dim/2.)
        return tf.random_normal(shape=size,stddev=xavier_stddev)

    # D network 
    def discriminator(self, x, y):
        # theta_D = [D_W1, D_W2, D_b1, D_b2]
        inputs = tf.concat(axis=1, values=[x,y])
        D_h1 = tf.nn.tanh(tf.matmul(inputs,self.D_W1)+self.D_b1)
        D_logit = tf.matmul(D_h1,self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    # G network
    def generator(self, z, y):
        # theta_G = [G_W1, G_W2, G_b1, G_b2]
        inputs = tf.concat(axis=1, values=[z,y])
        G_h1 = tf.nn.tanh(tf.matmul(inputs,self.G_W1)+self.G_b1)
        G_log_prob = tf.matmul(G_h1,self.G_W2)+self.G_b2
        G_prob = tf.nn.tanh(G_log_prob)
        return G_prob

    # noise generator
    def sample_Z(self,m,n):
        return np.random.randn(m,n)
    
    
    
