# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:33:31 2019

@author: khkim

[Perceptron]
다수의 신호를 입력으로 받아 하나의 신호를 출력

"""



def percep(x1, x2, w1, w2, theta):
    val = x1*w1 + x2*w2
    
    if val <= theta:
        return 0
    else:
        return 1
    
    
    
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    return percep(x1, x2, w1, w2, theta)
    


def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    return percep(x1, x2, w1, w2, theta)