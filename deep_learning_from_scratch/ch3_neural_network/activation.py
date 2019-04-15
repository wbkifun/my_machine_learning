# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:42:55 2019

@author: khkim

Activation functions for neural network
"""

import numpy as np



def step(x):
    return np.heaviside(x, 0)



def sigmoid(x):
    return 1/(1 + np.exp(-x))



def relu(x):
    '''
    Rectified Linear Unit
    '''
    return np.maximum(0, x)