# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:02 2023

@author: isdao
"""
import numpy as np

def sigmoid(x):
    """sigoid activation functions to be used in neural nets"""
    
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """ derivative of sigmoid functions for backprop"""
    
    return (np.exp(-x))/(1+np.exp(-x))**2

    
class NeuralNet():
    def __init__(self,dim): #instance defined by dimensions of NN
        
        #define dimensions of network
        #dim is list with dimensions of network: input data length,depth and widths, in that order.
        assert len(dim) == 3
        
        
        self.inputsize = dim[0]
        self.depth = dim[1]
        self.widths = dim[2]
        
        # assert there are as many widths as the depth of the network
        assert len(self.widths) ==  self.depth 
        
        self.outputsize = dim[2][-1]
        
        #define parameters of network
        #initialize list of weight matrices
        self.weight_mats = [np.random.randn(self.widths[i+1],
                                            self.widths[i]) 
                            for i in range(self.depth-1)]
        self.weight_mats.insert(0,np.random.randn(self.widths[0],
                                           self.inputsize))
        
        #assert len(self.depths) ==  len(self.weight_mats)
        
            
        
    def forward(self,data): 
        """Method to feed data forward through NN. Returns output"""
        #assert the input data can be used on this network
        assert len(data) == self.inputsize
        
        data = data.reshape((len(data),1))
        
        # initializaing activation list
        act = [0 for _ in range(self.depth)]
        weight_in = [0 for _ in range(self.depth)]
        
        for i in range(self.depth):
            if i == 0:
                weight_in[i] = np.dot(self.weight_mats[i],data)
                act[i] = sigmoid(weight_in[i])
            else:
                weight_in[i] = np.dot(self.weight_mats[i],act[i-1])
                act[i] = sigmoid(weight_in[i])
        
        return act,weight_in
    
    
    def cost(self,data,output):
        """calculates the cost for a given output"""
        assert len(output) == self.widths[-1]
        
        data = data.reshape((len(data),1))
        output = output.reshape((len(output),1))
        
        act, _ = self.forward(data)
        
        return np.sum(0.5*(act[-1] - output)**2)
    
    
    def gradient(self,data,output):
        
        data = data.reshape((len(data),1))
        output = output.reshape((len(output),1))
        
        
        act,weight_in = self.forward(data)
        
        
        deltas = [0 for _ in range(self.depth)]
        
        for i in range(-1,-self.depth-1,-1):
            
            if i == -1:
                deltas[i] = (act[i]-output)*sigmoid_prime(weight_in[i])
            else:
                deltas[i] = np.dot(self.weight_mats[i+1].transpose(),deltas[i+1])*sigmoid_prime(weight_in[i])
                
        grad_w = [np.dot(deltas[i+1],act[i].transpose()) for i in range(self.depth-1)]
                         
        grad_w.insert(0,np.dot(deltas[0],data.transpose()))
        
                
        return grad_w
    
    def update_grad(self,data,output,eta):
        
        grad_w = self.gradient(data,output)
        
        self.weight_mats = [w-eta*gw for w,gw in zip(self.weight_mats,grad_w)]
        
        return
    
    
    def train(self,data,output,eta,tol,maxit = 1000):
        """train the neural network with training input DATA 
        output, with a stepsize ETA to a tolerancr TOL"""
           
        costs = [self.cost(data,output)]
        
        while costs[-1] > tol:
                
            self.update_grad(data,output,eta)
            
            costs.append(self.cost(data,output))
            
        final_act, _ = self.forward(data)
        
        return final_act[-1], costs
        
    
    
       
            
            
        
        