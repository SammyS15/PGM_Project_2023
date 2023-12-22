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
print("oooh hello")

def applyppca(dataset_real, dataset_fakes, pca_num):
    # np.random.seed(42)
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

    def log_likelihood(W, var, S):
        C = W@W.T + np.identity(d)*var
        return -t/2*(d*np.log(2) + np.log(np.linalg.det(C)) + np.trace(np.linalg.pinv(C)@S))

    def log_likelihood2(W, var, x, z, mu, q):
        res = 0
        for i in range(t):
            res += -d/2*np.log(2*np.pi*var) + -np.linalg.norm(x[i] - W@z.T[i] - mu)**2/(2*var)
            res += -q/2*np.log(2*np.pi) - np.linalg.norm(z.T[i])**2/2

        return res

    # EM step
    while norm > np.e**-10: # 0.0001:
        
        # calculate M and its inverse
        M = W.T @ W + var * np.eye(pca_num)
        M_inv = np.linalg.pinv(M)
        
        # update W
        W_hat = S @ W @ np.linalg.pinv(var * np.eye(pca_num) + (M_inv @ W.T @ S @ W)) 
        
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

    ll = log_likelihood(W, var, S)

    ll2 = log_likelihood2(W, var, X_real, pca_X_real, mean, pca_num)

    return W, var, pca_X_real, pca_X_fake, score, coeff, ll, ll2

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

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import inv
from numpy import transpose as tr
# import ipdb
# https://github.com/cangermueller/ppca/blob/master/src/pca/ppca.py

class PPCA_Online(object):
    def __init__(self, q=2, sigma=1.0):
        self.q = q
        self.prior_sigma = sigma

    def fit(self, y, em=False):
        self.y = y
        self.p = y.shape[0]
        self.n = y.shape[1]
        if em:
            [self.w, self.mu, self.sigma] = self.__fit_em()
        else:
            [self.w, self.mu, self.sigma] = self.__fit_ml()

    def transform(self, y=None):
        if y is None:
            y = self.y
        [w, mu, sigma] = [self.w, self.mu, self.sigma]
        m = tr(w).dot(w) + sigma * np.eye(w.shape[1])
        m = inv(m)
        x = m.dot(tr(w)).dot(y - mu)
        return x

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform()

    def transform_infers(self, x=None, noise=False):
        if x is None:
            x = self.transform()
        [w, mu, sigma] = [self.w, self.mu, self.sigma]
        y = w.dot(x) + mu
        if noise:
            for i in range(y.shape[1]):
                e = np.random.normal(0, sigma, y.shape[0])
                y[:, i] += e
        return y

    def __ell(self, w, mu, sigma, norm=True):
        m = inv(tr(w).dot(w) + sigma * np.eye(w.shape[1]))
        mw = m.dot(tr(w))
        ll = 0.0
        for i in range(self.n):
            yi = self.y[i][:, np.newaxis]
            yyi = yi - mu
            xi = mw.dot(yyi)
            xxi = sigma * m + xi.dot(tr(xi))
            ll += 0.5 * np.trace(xxi)
            if sigma > 1e-5:
                ll += (2 * sigma)**-1 * float(tr(yyi).dot(yyi))
                ll -= sigma**-1 * float(tr(xi).dot(tr(w)).dot(yyi))
                ll += (2 * sigma)**-1 * np.trace(tr(w).dot(w).dot(xxi))
        if sigma > 1e-5:
            ll += 0.5 * self.n * self.p * np.log(sigma)
        ll *= -1.0
        if norm:
            ll /= float(self.n)
        return ll

    def __fit_em(self, maxit=20):
        w = np.random.rand(self.p, self.q)
        mu = np.mean(self.y, 1)[:, np.newaxis]
        sigma = self.prior_sigma
        ll = self.__ell(w, mu, sigma)

        yy = self.y - mu
        s = self.n**-1 * yy.dot(tr(yy))
        for i in range(maxit):
            m = inv(tr(w).dot(w) + sigma * np.eye(self.q))
            t = inv(sigma * np.eye(self.q) + m.dot(tr(w)).dot(s).dot(w))
            w_new = s.dot(w).dot(t)
            sigma_new = self.p**-1 * np.trace(s - s.dot(w).dot(m).dot(tr(w_new)))
            ll_new = self.__ell(w_new, mu, sigma_new)
            # print("{:3d}  {:.3f}".format(i + 1, ll_new))
            w = w_new
            sigma = sigma_new
            ll = ll_new
        return (w, mu, sigma)

    def __fit_ml(self):
        mu = np.mean(self.y, 1)[:, np.newaxis]
        [u, s, v] = np.linalg.svd(self.y - mu)
        if self.q > len(s):
            ss = np.zeros(self.q)
            ss[:len(s)] = s
        else:
            ss = s[:self.q]
        ss = np.sqrt(np.maximum(0, ss**2 - self.prior_sigma))
        w = u[:, :self.q].dot(np.diag(ss))
        if self.q < self.p:
            sigma = 1.0 / (self.p - self.q) * np.sum(s[self.q:]**2)
        else:
            sigma = 0.0
        return (w, mu, sigma)

def apply_ppca_online(dataset_real, dataset_fakes, pca_num):
    ov = PPCA_Online(q = pca_num)
    ov.fit(dataset_real.T, em=True)
    pcs_ov_X_real = ov.transform()
    pcs_ov_X_fakes = ov.transform(dataset_fakes.T)
    score_ov = ov.transform().T[:,0:2]
    coeff_ov = ov.w.values[:,0:2]
    
    return ov.w, ov.sigma, pcs_ov_X_real, pcs_ov_X_fakes, score, coeff

