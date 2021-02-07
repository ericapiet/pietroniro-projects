# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:36:12 2021

@author: erica
"""

#%% Import
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


#%% Load Data
filename = os.path.abspath(
    r"C:\Users\erica\Documents\syde552\Assignment1\assignment1-data\regression-data.pkl")
data = pickle.load(open(filename, 'rb'), encoding='latin1')

test_x = data["testx"][0]
test_y = data["testy"][0]
train_x = data["trainx"][0]
train_y = data["trainy"][0]


#%% Definitions
SIGMA = 0.2
MU = 1
lam = [10e-8, 10e-5, 10e-2, 10]
lam_str = ["10e-8", "10e-5", "10e-2", "10"]


def gaussian(x, b):
    return MU * np.exp(-0.5*(x-b)**2/SIGMA**2)


phi = np.linspace(-1, 1, 20)
gauss_x = np.linspace(-1, 1, 100)
 
    
def Regression_model(x_in, y_in):
    i_len = len(x_in)
        
    # Build F matrix
    F = np.zeros((i_len, i_len))
    
    for i in range(i_len):
        for ii in range(i_len):
            F[i][ii] = gaussian(x_in[i], phi[ii])
            
    T = y_in
    W = []
    I = np.identity(i_len)
    
    for i in range(len(lam)):
        lam_i = lam[i]
    
        # Solve for weights
        W.append(
            (np.linalg.inv(F.transpose().dot(F) + lam_i*I)).dot(F.transpose().dot(T)))
    
    x_reg = np.arange(-1, 1.05, 0.05)
    y_reg = np.zeros(len(x_reg))
    rms = np.zeros(len(lam))
    y_pred = np.zeros(len(y_in))
    
    #plot input data
    plt.figure()
    plt.scatter(x_in, y_in)
    
    # Gaussian regression equation
    for k in range(len(lam)):
        w = W[k]
        for i in range(len(x_reg)):
            y_reg[i] = sum(
                [w[j]*gaussian(x_reg[i], phi[j]) for j in range(i_len)])
        
        # calculate mean squared error
        for i in range(i_len):
            
            y_pred[i] = sum(
                [w[j]*gaussian(x_in[j], phi[j]) for j in range(i_len)])
            
        rms[k] = np.sqrt(
            sum([(y_pred[j] - x_in[j])**2 for j in range(i_len)])/i_len)
        
        plt.plot(x_reg, y_reg, label="lambda = %s"%lam_str[k])
        plt.title('Q2-b Least Squares Regression')
        plt.legend()
    
    # plot RMS error
    plt.figure(5)
    plt.plot(np.log(lam), rms, label="training")
    plt.title('Q2-c RMS error data')
    plt.xlabel('ln(lambda)')
    plt.ylabel('E(w)')
    plt.legend()
    
    return W
    

#%% Plot

# plot set of basis functions
plt.figure(1)
for phi_j in phi:
    plt.plot(gauss_x, gaussian(gauss_x, phi_j))
    plt.title('Q2-a Basis Functions')

# plot training input data

weights = Regression_model(train_x, train_y)


# plot test data

# Gaussian regression equation
rms = np.zeros(len(lam))
i_len = len(test_x)
y_pred = np.zeros(i_len)

for k in range(len(lam)):
    w = weights[k]
    
    # calculate mean squared error
    for i in range(i_len):
        
        y_pred[i] = sum(
            [w[j]*gaussian(test_x[j], phi[j]) for j in range(i_len)])
        
    rms[k] = np.sqrt(
        sum([(y_pred[j] - test_x[j])**2 for j in range(i_len)])/i_len)
    
# plot RMS error
plt.figure(5)
plt.plot(np.log(lam), rms, label="test")
plt.title('Q2-c RMS error test data')
plt.xlabel('ln(lambda)')
plt.ylabel('E(w)')
plt.legend()


#%% Test Data


    
    