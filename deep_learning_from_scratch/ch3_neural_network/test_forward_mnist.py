# -*- coding: utf-8 -*-

from PIL import Image
from numpy.testing import assert_array_almost_equal as aa_equal
import numpy as np
import matplotlib.pyplot as plt

from forward_mnist import load_data, load_network
from forward_mnist import check_weights, check_weights_batch



def test_forward():
    x_test, t_test = load_data()
    network = load_network()

    accuracy = check_weights(network, x_test, t_test)
    aa_equal(accuracy, 0.9352, 4)  # 93.52 %



def test_forward_batch():
    x_test, t_test = load_data()
    network = load_network()

    accuracy = check_weights_batch(network, x_test, t_test)
    aa_equal(accuracy, 0.9352, 4)



def plot_digit_img(images, labels, idx):
    '''
    plot an digit image
    '''
    print("label={}".format(labels[idx]))
    img = images[idx].reshape(28, 28)

    #pil_img = Image.fromarray(np.uint8(img))
    #pil_img.show()

    plt.imshow(img)
    plt.show()



if __name__ == '__main__':    
    images, labels = load_data(norm=False)
    plot_digit_img(images, labels, idx=0)
