# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:02 2023

@author: isdao
"""

class NeuralNet():
    def __init__(self,dim: tuple):
        #define dimensions of network
        assert len(dim) == 4
        self.inputsize = tuple[0]
        self.depth = tuple[1]
        self.width = tuple[2]
        self.outputsize = tuple[3]
        
        #define parameters of network
        
        