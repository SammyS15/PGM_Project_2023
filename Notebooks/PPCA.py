get_ipython().run_line_magic('matplotlib', 'inline')
from read_yse_ztf_snana_dir import read_YSE_ZTF_snana_dir
from matplotlib.backends.backend_pdf import PdfPages


import glob
import sncosmo
import light_curve
from light_curve import VillarFit
import extinction
from extinction import fm07, apply, remove
import statistics as st

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#plt.style.use('fig_publication.mplstyle')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # Load fakes (sims) DF and DR4 DF

# In[2]:


import os
os.getcwd()

def applyppca(dataset_real, dataset_fakes, pca_num):
    np.random.seed(42)
    # convert the datasets to numpy arrays (I haven't included the fakes in the EM step yet)
    X_real = dataset_real.to_numpy() 
    X_fakes = dataset_fakes.to_numpy()
    
    t,d = X_real.shape
    
    # initialize W with random numbers, shape (d, pca_num)
    W = np.random.randn(d, pca_num)
    W_hat = np.zeros((d, pca_num))
    
    # calculate the mean across all data points
    mean = np.mean(X_real, axis=0) 
    
    # calculate S
    S = (1/t)*(X_real - mean[np.newaxis,:]).T @ (X_real - mean[np.newaxis,:])
    
    # initialize the variance to 1
    var = 1

    # calculate the norm squared between current and past weight matrix as a convergence metric
    norm = np.linalg.norm(W-W_hat)**2
    
    # EM step
    while norm > 0.0001:
        
        # calculate M and its inverse
        M = W.T @ W + var * np.eye(pca_num)
        M_inv = np.linalg.pinv(M)
        
        # update W
        W_hat = S @ W @ np.linalg.pinv(var * np.eye(pca_num) +(M_inv @ W.T @ S @ W)) 
        
        # update the variance with new W
        var = (1/d)*np.trace(S - S @ W @ M_inv @ W_hat.T)
        
        # update the norm squared
        norm = np.linalg.norm(W-W_hat)**2
        
        # set current W to old W
        W = W_hat

    # apply the derived W to the centred data 
    pca_X_real = W.T @ (X_real - mean[np.newaxis,:]).T 
    mean_fake = np.mean(X_fakes, axis=0)
    pca_X_fake = W.T @ (X_fakes - mean_fake[np.newaxis,:]).T
    # calculate the score and coeffs from the pca application and W
    score = pca_X_real.T[:,0:2]
    coeff = np.transpose(W.T[0:2, :])

    return W, var, pca_X_real, pca_X_fake, score, coeff

def applypca(dataset_real, dataset_fakes, pca_num):
    X_real = dataset_real
    X_fakes = dataset_fakes
    
    ss = StandardScaler()
    scaled_X_real = ss.fit_transform(X_real)
    pca = PCA(pca_num) # PCA(n_components=8)
    
    # apply same PCA to both real and fakes data
    pcs_X_real = pca.fit_transform(scaled_X_real) #Fit the model with X and apply the dimensionality reduction on X.
    pcs_X_fakes = pca.transform(ss.transform(X_fakes)) #Apply dimensionality reduction to X.
    
    score = pcs_X_real[:,0:2]
    coeff = np.transpose(pca.components_[0:2, :])

    return pca, pcs_X_real, pcs_X_fakes, score, coeff