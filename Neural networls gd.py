# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:24:18 2021

@author: saini
"""
#Implementation of Gradient Descent
#Define Sigmoid Function
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
#define derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1- sigmoid(x))
input = np.array([0.5 ,0.8])
y = .8
weights = np.array([-0.8,0.6])
learning_rate= 0.5
function = np.dot(input ,weights)
print(function)
output = sigmoid(function)
print(output)    
error = y-output
print(error)
output_grad = sigmoid_der(output)
print(output_grad)
error_term =error *output_grad
print(error_term)
fnal = learning_rate*error_term*input
print(fnal)
