# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:12:08 2021

@author: erica
"""
#%% Import
import numpy as np
import os
import pickle
import pandas as pd


#%% Load Data
filename = os.path.abspath(
    r"C:\Users\erica\Documents\syde552\Assignment1\assignment1-data\backprop-data.pkl")
data = pickle.load(open(filename, 'rb'), encoding='latin1')

vectors = data['vectors']
labels = data['labels'].astype(np.int)
vectors = data['vectors']

#put in dataframe
training_df = pd.DataFrame(columns=["f_1", "f_2", "labels", "outputs"])
training_df.f_1 = [x for x in vectors[0]]
training_df.f_2 = [x for x in vectors[1]]
training_df.labels = labels

#%% Definitions

GAMMA = 0.01

# output layer
g_output = lambda a: 1/(1 + np.e**-a)
g_output_deriv = lambda a: g_output(a)*(1-g_output(a))

# hidden layer
g_hidden = lambda b: np.tanh(b)
g_hidden_deriv = lambda b: np.sech(b)**2


y_fun = lambda a: g_hidden(a)
z_fun = lambda b: g_output(b)

E = lambda y, t: 0.5*(y - t)**2

delta_w_output = lambda y, z, t, b: GAMMA*(t - z)*g_output_deriv(b)*y

delta_w_hidden = lambda x, z, t, w, a, b: GAMMA*(t - z)*g_hidden_deriv(b)*w*g_hidden_deriv(a)*x

hidden_w = [0, 0]
output_w = [0, 0]




