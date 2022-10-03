# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:28:50 2022

@author: neelkh
"""

from multilayer_perceptron import Multilayer_Perceptron
import numpy as np


for seeed in range(4):
    
    # print('Multilayer perceptron with weights : \n')
    np.random.seed(seeed)
    
    w1 = np.random.ranf((2,3))
    b1 = np.random.ranf((1,3))
    
    
    w2 = np.random.ranf((3,2))
    b2 = np.random.ranf((1,2))
    
    
    w3 = np.random.ranf((2,2))
    b3 = np.random.ranf((1,2))
    
    mlp = Multilayer_Perceptron(w1, b1, w2, b2, w3, b3)
    
    x = np.array([1,1])
    
    print('\n Multilayer perceptron with weights : \n')
    print('w1 \n', w1, '\n', 'w2 \n', w2, '\n', 'w3 \n', w3)

    print('\n Multilayer perceptron with bias : \n')
    print('b1 \n', b1, '\n', 'b2 \n', b2, '\n', 'b3 \n', b3)
    
    y = mlp.forward(x)

    print('\n Y is :', y, '\n ----------------------')