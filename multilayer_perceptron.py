# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class Multilayer_Perceptron:
    
    def __init__(self, w1, b1, w2, b2, w3, b3):
        
        
        self.net={}        
        
        self.net['w1'] = w1
        self.net['b1'] = b1
        
        self.net['w2'] = w2
        self.net['b2'] = b2
        
        self.net['w3'] = w3
        self.net['b3'] = b3        
        
    
    def sigmoid(self, a):
        
        return 1/(1+np.exp(-a))
    
    
    def forward(self, x):
        
        w1, w2, w3 = self.net['w1'] ,self.net['w2'] ,self.net['w3']
        b1, b2, b3 = self.net['b1'] ,self.net['b2'] ,self.net['b3']
        
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        
        a3 = np.dot(z2,w3) + b3
        
        return a3