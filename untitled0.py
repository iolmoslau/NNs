# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:02 2023

@author: isdao
"""
import numpy as np

class NeuralNet():
    def __init__(self,dim: tuple):
        #define dimensions of network
        assert len(dim) == 4
        self.inputsize = tuple[0]
        self.depth = tuple[1]
        self.width = tuple[2]
        self.outputsize = tuple[3]
        
        #define parameters of network
        #initialize list of weight matrices
        weight_mats = []
        
        for _ in range(self.depth):
            weight_mats.append(np.random.randn(self.width,
                                               self.inputsize))
        
    def 
            
            
            
        
        