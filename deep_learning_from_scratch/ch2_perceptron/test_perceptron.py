# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:35:36 2019

@author: khkim
"""

from numpy.testing import assert_equal as equal
from perceptron import AND, NAND


def test_AND():
    equal(AND(0, 0), 0)
    equal(AND(1, 0), 0)
    equal(AND(0, 1), 0)
    equal(AND(1, 1), 1)
    
    
def test_NAND():
    equal(NAND(0, 0), 1)
    equal(NAND(1, 0), 1)
    equal(NAND(0, 1), 1)
    equal(NAND(1, 1), 0)