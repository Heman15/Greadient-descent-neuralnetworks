# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:24:18 2021

@author: saini
"""
#Implementation of Gradient Descent
#Define Sigmoid Function
import numpy as np #created sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#define derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1- sigmoid(x))
input = np.array([0.5 ,0.8]) #input values
y = .8  #target value
weights = np.array([-0.8,0.6])  #defining intial weights
learning_rate= 0.5 #defining learning  rate
function = np.dot(input ,weights) #get f(Y) = x1w1 +x2w2 +x3w3 +x4w4 + .......
print(function) 
output = sigmoid(function) #geting output
print(output)    
error = y-output #calculating error between target value and outputvalue
print(error)
output_grad = sigmoid_der(output)    #applying activation function
print(output_grad)
error_term =error *output_grad  #cal. error term
print(error_term)
fnal = learning_rate*error_term*input       #final output gd step
print(fnal)
