# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:44:43 2019

@author: khkim
"""
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
import numpy as np
import matplotlib.pyplot as plt

from gradient import numerical_diff
from gradient import numerical_gradient
from gradient import gradient_descent



def test_numerical_diff():
    f = lambda x: 3*x*x + 2
    x = np.linspace(-2, 2, 1000)
    ret = numerical_diff(f, x)
    
    df = lambda x: 6*x
    aa_equal(ret, df(x), 10)



def test_numerical_gradient():
    f = lambda x: x[0]*x[0] + x[1]*x[1]
    df = lambda x: [2*x[0], 2*x[1]]
    x = np.array([3,4], 'f8')  # single coordinates
    
    ret = numerical_gradient(f, x)
    aa_equal(ret, df(x), 10)
    

    
def test_numerical_gradient_array():
    f = lambda x: 2*x[0]*x[0] + 3*x[1]*x[1] + 4*x[2]*x[2]
    x = 4*np.random.rand(1000, 3) - 2  # [-2, 2), array of coordinates
    
    ret = numerical_gradient(f, x)
    
    ref = np.zeros_like(x, 'f8')
    ref[:,0] = 4*x[:,0]
    ref[:,1] = 6*x[:,1]
    ref[:,2] = 8*x[:,2]
    aa_equal(ret, ref, 10)
    
    
    
def test_gradient_descent():
    f = lambda x: x[0]*x[0] + x[1]*x[1]
    
    init_x = np.array([-3., 4.])
    ret = gradient_descent(f, init_x, lr=0.1, num_step=100)
    assert np.all(np.abs(ret) < 1e-8)