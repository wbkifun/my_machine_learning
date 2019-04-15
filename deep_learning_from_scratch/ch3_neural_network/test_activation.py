# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:53:12 2019

@author: khkim
"""
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
import numpy as np
import matplotlib.pyplot as plt

from activation import step, sigmoid, relu



def test_step():
    x = np.random.rand(100) + 1e-10  # [0,1) + 1e-10
    equal(step(0), 0)
    a_equal(step(x), 1)
    a_equal(step(-x), 0)
    
    
    
def test_relu():
    x = np.random.rand(100) + 1e-10  # [0,1) + 1e-10
    equal(relu(0), 0)
    a_equal(relu(x), x)
    a_equal(relu(-x), 0)
    
    
    
def plot(func_name):
    x = np.arange(-5, 5, 0.1)
    y = globals()[func_name](x)
    plt.plot(x, y)
    
    ymax = 5.1 if func_name == 'relu' else 1.1
    plt.ylim(-0.1, ymax)
    plt.show()
        
    

if __name__ == '__main__':
    print("1:Step, 2:Sigmoid, 3:ReLU")
    func_id = int(input("Select an activation function: "))
    
    func_name = {1:'step', 2:'sigmoid', 3:'relu'}[func_id]
    print("Plot the {} function".format(func_name))
    plot(func_name)