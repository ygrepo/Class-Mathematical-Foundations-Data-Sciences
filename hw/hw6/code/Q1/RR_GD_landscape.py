#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:26:31 2020

@author: carlos
"""

import matplotlib
import numpy as np

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.colors

plt.close('all')
np.random.seed(1234)


# Gaussian density
def gaussian(x, y, Sigma, mean):
    invSigma = np.linalg.inv(Sigma)
    return (np.exp(-(invSigma[0, 0] * (x - mean[0]) ** 2
                     + 2 * invSigma[0, 1] * (x - mean[0]) * (y - mean[1])
                     + invSigma[1, 1] * (y - mean[1]) ** 2) / 2) / (2 * np.pi *
                                                                    np.sqrt(np.linalg.det(Sigma))))


def gradient_descent(X, y, alpha, k, reg_param):
    # WRITE YOUR CODE FOR GRADIENT DESCENT HERE
    runner = (1 - alpha * reg_param) * np.identity(2)
    runner = runner - alpha * np.matmul(X, X.T)
    prod_runner = np.identity(2)
    for i in range(1, k):
        prod_runner = np.matmul(runner, prod_runner)
        prod_runner = prod_runner + np.identity(2)
    prod_runner = prod_runner + np.identity(2)
    Xy = X @ y
    beta_vec = prod_runner @ Xy
    beta_vec = alpha * beta_vec
    beta_vec = beta_vec[..., np.newaxis]
    return beta_vec


def compute_bias_covariance(U, S, k, alpha, reg_param, beta_true, std):
    diag_bias = np.identity(2)
    diag_sigma = np.identity(2)
    for i in range(2):
        s2 = S[i] ** 2
        t = s2 + reg_param
        tau = 1 - ((1 - alpha * t) ** k)
        print(f"tau:{tau}")
        diag_bias[i, i] = (tau * s2) / t
        diag_sigma[i, i] = diag_bias[i, i] * tau
        diag_sigma[i, i] /= t

    bias = U.T @ beta_true
    bias = diag_bias @ bias
    bias = U @ bias

    sigma = diag_sigma @ U.T
    sigma = U @ sigma
    sigma = (std**2) * sigma
    return bias[..., np.newaxis], sigma


X = np.array([[0.1, 0], [0, 1]])
U, S, V = np.linalg.svd(X, full_matrices=False)
print(S)

beta_true = np.array([1, 1])

data_clean = X.T @ beta_true

# Regularization parameter
reg_param = 0.05
# Standard deviation of the noise
noise_std = 0.2
# Step size for gradient descent
alpha = 0.2
# Number of iterations for gradient descent
k_vec = [3, 50, 500]
# Noise realizations
n_noise = 200
noise = noise_std * np.random.randn(2, n_noise)
# Auxiliary vectors for plotting
v1 = np.linspace(-3, 6, 100)
v2 = np.linspace(-2, 4, 100)
V1, V2 = np.meshgrid(v1, v2)

for ind, k in enumerate(k_vec):
    est_betas = np.zeros((2,n_noise))

    for ind_noise in range(n_noise):
        z = noise[:,ind_noise]
        y = data_clean + z
        betas = gradient_descent(X,y,alpha,k,reg_param)
        est_betas[:,ind_noise] = betas[:,-1]
    fig = plt.figure(figsize = (9,6))
    plt.scatter(est_betas[0,:],est_betas[1,:],color="teal",s=6)
    plt.plot(1,1,'o',color='red',markersize=6)
    ax = plt.axes()
    ax.text(1-0.1,1 - 0.55,r'$\mathbf{\beta_{true}}$',
    fontsize=20,color='red')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r'$\beta[1]$', fontsize=18,labelpad=10)
    h = plt.ylabel(r'$\beta [2] $', fontsize=18,labelpad=15)
    h.set_rotation(0)
    plt.xlim([-3,6])
    plt.ylim([-2,4])
    plt.savefig('RR_GD_scatterplot'+str(ind)+'.pdf',bbox_inches='tight')


    # WRITE YOUR CODE TO GENERATE THE MEAN bias_GD AND COVARIANCE MATRIX
    # Sigma OF THE COEFFICIENT ESTIMATOR HERE
    bias_GD, Sigma = compute_bias_covariance(U, S, k, alpha, reg_param, beta_true, noise_std)
    print(bias_GD)
    print(Sigma)

    Z_gauss = gaussian(V1, V2, Sigma, bias_GD)
    Z_gauss[Z_gauss == 0] = 1e-20
    Z_gauss[Z_gauss < 1e-20] = 1e-20
    fig = plt.figure(figsize=(11, 6))
    lev_exp = np.arange(-15,
                        np.ceil(np.log10(Z_gauss.max()) + 1))
    levs = np.power(10, lev_exp)
    # levs = np.insert(levs, 0,1e-10)
    CS = plt.contourf(V1, V2, Z_gauss, levels=levs, norm=matplotlib.colors.LogNorm(),
                      cmap='bone', extend='min');
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    ax = plt.axes()
    plt.plot(bias_GD[0], bias_GD[1], 'o', color='green', label=r'$\mathbf{\beta_{bias}}$')
    plt.plot(1, 1, 'o', color='red', label=r'$\mathbf{\beta_{true}}$')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r'$\beta[1]$', fontsize=18, labelpad=10)
    h = plt.ylabel(r'$\beta [2] $', fontsize=18, labelpad=15)
    h.set_rotation(0)
    plt.legend(fontsize=20, framealpha=1)
    plt.savefig('RR_GD_scatterplot_distribution' + str(ind) + '.pdf',
                bbox_inches='tight')
var1 = noise_std**2 * (1**2) / ((1 + reg_param)**2)
var2 = noise_std**2 * (0.1**2) / ((0.1 + reg_param)**2)

print(var1, var2)