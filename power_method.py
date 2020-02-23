import numpy as np

(n, d) = 400, 8
covariance = np.random.uniform(-1, 1, (d, d))
covariance = np.dot(covariance.T, covariance)
X = np.random.multivariate_normal(np.zeros(d), covariance, n)

import numpy as np
from numpy import linalg as LA


def power_method(M, x_0, steps=50):
    """
    Perform the power method on M starting at x_0 to find the largest eigenvector
    Args:
      M: a numpy array with shape (D, D).  matrix whose eigenvector is to be computed
      x_0: a numpy array with shape (D,).  starting vector for power method
      steps: number of iteration to run power_method
    Returns:
      approximate eigenvector with unit Euclidean norm, numpy array with shape (D,)
    """
    x = x_0
    x = x/LA.norm(x)
    for i in range(steps):
        x = np.matmul(x,M)/LA.norm(np.matmul(x,M))
    return x


def deflate(M, lamb, q):
    """
    Perform deflation on M using approximate eigenvalue and eigenvector
    Args:
      M: a numpy array with shape (D, D).  matrix to be deflated
      lamb: float.  eigenvalue to be removed
      q: a numpy array with shape (D,).  unit-norm eigenvector to be removed
    Returns:
      deflated matrix, numpy array with shape (D, D)
    """
    Mp = M
    s = q.shape[0]
    q = q.reshape(s,1)
    Mp = Mp - lamb*(np.matmul(q,np.transpose(q))/np.matmul(np.transpose(q),q))
    return Mp


def pca(X, k):
    """
    Calculate the PCA of matrix X with k dimensions
    Args:
      X: a numpy array with shape (N, D).  matrix whose PCA is to be computed
      k: float.  number of dimensions for the PCA (<D)
    Returns:
      PCA matrix with , numpy array with shape (k, D)
    """
    D = X.shape[1]
    PCA = np.zeros((k, D))
    # YOUR CODE HERE
    M = np.matmul(np.transpose(X),X)
    x_0 = np.random.rand(D)
    eigen_vec = np.zeros((D,k))
    v_k = power_method(M,x_0)
    lamb_k = v_k.dot(M.dot(v_k))
    return lamb_k


