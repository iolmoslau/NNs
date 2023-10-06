# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:02 2023

@author: isdao
"""
import numpy as np

class NeuralNet():
    def __init__(self,dim,data_in): #instance defined by dimensions of NN and input data
        # define input data
        self.data = data_in
        
        #define dimensions of network
        assert len(dim) == 4
        assert len(dim[0]) == 1
        
        self.inputsize = dim[0]
        self.depth = dim[1]
        self.widths = dim[2]
        
        assert len(self.widths) ==  self.depth
        
        self.outputsize = dim[3]
        
        #define parameters of network
        #initialize list of weight matrices
        self.weight_mats = [np.random.randn(self.widths[i+1],
                                            self.widths[i]) 
                            for i in range(self.depth-1)]
        self.weight_mats.insert(0,np.random.randn(self.withds[0],
                                           self.inputsize))
        
        assert len(self.depths) ==  len(self.weight_mats)
        
            
        
    def forward(self): 
        assert self.data.shape[0] == self.weight_mats[0]
        
        act = [0 for _ in range(self.depth)]
       
        
        for i in range(self.depth):
            if i == 0:
                act[i] = np.dot(self.weight_mats[i],self.data)
            else:
                act[i] = np.dot(self.weight_mats[i],act[i-1])
        
        return act[-1]
    
    
       
            
            
        
        