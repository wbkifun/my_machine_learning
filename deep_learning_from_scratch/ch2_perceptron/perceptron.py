# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:33:31 2019

@author: khkim

[Perceptron]
다수의 신호를 입력으로 받아 하나의 신호를 출력

"""



def percep(x1, x2, w1, w2, bias):
    val = x1*w1 + x2*w2 + bias
    
    return 0 if val <= 0 else 1
    
    
    
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, -0.7
    return percep(x1, x2, w1, w2, theta)
    


def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, 0.7
    return percep(x1, x2, w1, w2, theta)



def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, -0.4
    return percep(x1, x2, w1, w2, theta)



def XOR(x1, x2):
    '''
    Multi-layer perceptron
    '''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    out = AND(s1, s2)
    
    return out