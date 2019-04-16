# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:22:36 2019

@author: khkim

forward propagation with the MNIST dataset
"""

from PIL import Image
import pickle
import numpy as np

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



def plot_img(images, labels, idx):
    '''
    plot an digit image
    '''
    print("label={}".format(labels[idx]))
    img = images[idx].reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



class NeuralNetwork:
    def __init__(self):
        self.x_test, self.t_test = load_data()
            
    
    def forward(self, x):
        net = self.network
        
        W1, W2, W3 = net['W1'], net['W2'], net['W3']
        b1, b2, b3 = net['b1'], net['b2'], net['b3']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        
        return y
    
    
    def check_weights(self):        
        x_test, t_test = self.x_test, self.t_test
                
        with open('sample_weight.pkl', 'rb') as f:
            self.network = pickle.load(f)
            
        accuracy_cnt = 0
        for x, t in zip(x_test, t_test):
            y = self.forward(x)
            p = np.argmax(y)  # index of the most probility element
            if p == t:
                accuracy_cnt += 1
                
        print("Accuracy: {:2.2f} %".format(accuracy_cnt/len(x_test)*100))            
        
        
    
if __name__ == '__main__':    
    #images, labels = load_data(norm=False)
    #plot_img(images, labels, idx=0)
    
    nn= NeuralNetwork()
    nn.check_weights()