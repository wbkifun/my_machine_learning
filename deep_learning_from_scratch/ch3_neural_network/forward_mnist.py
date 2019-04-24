# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:22:36 2019

@author: khkim

forward propagation with the MNIST dataset
"""

from os.path import abspath, dirname, join
import sys
import pickle
import numpy as np

current_dir = dirname(abspath(__file__))
common_dir = join(dirname(current_dir), 'common')
sys.path.append(common_dir)
from mnist import load_mnist
from activation import sigmoid, softmax



def load_data(norm=True):
    '''
    x: image 28x28 pixel
    t: label 0~9
    '''
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=norm, flatten=True, one_hot_label=False)
            
    return x_test, t_test



def load_network():
    '''
    load the pre-learned weights
    '''
    sample_weight_fpath = join(dirname(current_dir), 'data', 'sample_weight.pkl')

    with open(sample_weight_fpath, 'rb') as f:
        network = pickle.load(f)

    return network
            
    

def forward(network, x):
    net = network
    
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
    
    

def check_weights(network, x_test, t_test):        
    accuracy_cnt = 0
    for x, t in zip(x_test, t_test):
        y = forward(network, x)
        p = np.argmax(y)  # index of the most probility element
        if p == t:
            accuracy_cnt += 1
            
    accuracy = accuracy_cnt/len(x_test)
    #print("Accuracy: {:2.2f} %".format(accuracy*100))            

    return accuracy
            
    

def check_weights_batch(network, x_test, t_test):        
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x_test), batch_size):
        x_b = x_test[i:i+batch_size]
        y_b = forward(network, x_b)
        p_b = np.argmax(y_b, axis=1)  # index of the most probility element
        accuracy_cnt += np.sum(p_b == t_test[i:i+batch_size])
            
    accuracy = accuracy_cnt/len(x_test)
    #print("Accuracy(batch): {:2.2f} %".format(accuracy*100))            

    return accuracy
