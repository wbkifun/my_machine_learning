# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:01:00 2019

@author: khkim
"""

import numpy as np



def numerical_diff(f, x):
    '''
    f: function
    x: 1d, scalar
    '''
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)
    
    

def numerical_gradient(f, x):
    '''
    f: N-d function
    x: N-d, scalar
    '''    
    h = 1e-4
    d = x.shape[-1]  # N-dimension
    im = np.eye(d)   # identity matrix
    grad = np.zeros_like(x, 'f8')
    
    if x.ndim == 1:  # single coordinates
        for k in range(d):
            hh = h*im[k]  # one-hot array        
            grad[k] = (f(x + hh) - f(x - hh))/(2*h)
            
    elif x.ndim == 2:  # array of coordinates
        for i in range(x.shape[0]):
            for k in range(d):
                hh = h*im[k]  # one-hot array        
                grad[i,k] = (f(x[i] + hh) - f(x[i] - hh))/(2*h)

    return grad
        
        

def gradient_descent(f, init_x, lr=0.01, num_step=100):
    x = init_x
    
    for i in range(num_step):
        grad = numerical_gradient(f, x)
        x -= lr*grad
        
    return x
    
    
    
'''
def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        
    return grad



def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad



def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad
'''