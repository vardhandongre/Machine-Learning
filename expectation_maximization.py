# Author: Vardhan Dongre
import scipy as sp
from scipy.stats import multivariate_normal
import numpy as np

def e_step(x, pi, mu, sigma):
    """
    @param      x      Samples represented as a numpy array of shape (n, d)
    @param      pi     Mixing coefficient represented as a numpy array of shape
                       (k,)
    @param      mu     Mean of each distribution represented as a numpy array of
                       shape (k, d)
    @param      sigma  Covariance matrix of each distribution represented as a
                       numpy array of shape (k, d, d)
    @return     The "soft" assignment of shape (n, k)
    """
    k = mu.shape[0]
    n, d = x.shape
    a = np.asmatrix(np.empty((n,k), dtype = float))
    for i in range(n):
        p1 = 0
        for j in range(d):
            aj = sp.stats.multivariate_normal.pdf(x[i,:], mu[j,:], sigma[j,:].A1)*pi(j)
            p1 += aj
            a[i,j] = aj
        a[i,:] /= p1
    return a


def m_step(x, a):
    """
    @param      x     Samples represented as a numpy array of shape (n, d)
    @param      a     Soft assignments of each sample, represented as a numpy
                      array of shape (n, k)
    @return     A tuple (pi, mu, sigma), where
                - pi is the mixing coefficient represented as a numpy array of
                shape (k,)
                - mu is the mean of each distribution represented as a numpy
                array of shape (k, d)
                - sigma is the covariance matrix of each distribution
                represented as a numpy array of shape (k, d, d)
    """
    n, k = a.shape
    d = x.shape[1]
    mu = np.asmatrix(np.random.random((k, d)))
    pi = np.ones(k)/k
    sigma = np.array([np.asmatrix(np.identity(d)) for i in range(k)])
    for j in range(k):
        G = a[:,j].sum()
        pi[j] = 1/n*G
        mu_j = np.zeros(d)
        sigma_j = np.zeros((d,d))
        for i in range(n):
            mu_j += (x[i,:]*a[i,j])
            sigma_j += a[i, j] * np.matmul(np.transpose(x[i, :] - mu[j, :]), (x[i, :] - mu[j, :]))
        mu[j] = mu_j / G
        sigma[j] = sigma_j / G
    return (pi, mu, sigma)


