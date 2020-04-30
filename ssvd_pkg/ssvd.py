# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la

def sign_new(x):
    """
    vectorized version of finding signals
    """
    y = x.copy()
    y[y!=0] = y[y!=0]/abs(y[y!=0])
    return y

def pos_new(x):
    """
    vectorized version of finding max(x, 0)
    """
    y = x.copy()
    y[y<0] = 0
    return y

def BIC_new(x, Y, lambd, var_ols):
    """
    vectorized version of computing BIC
    """
    N = Y.shape[0]
    beta_ols = la.inv(x.T@x) @ x.T @ Y
    beta_lasso = beta_ols * pos_new(1-lambd/beta_ols)
    Y_hat = x @ beta_lasso
    df = sum(beta_lasso != 0)
    err = sum((Y_hat - Y) ** 2)
    return err/(var_ols*N) + np.log(N)*df/N

def find_lambda_new(X, u, w_2):
    """
    vectorized version of finding lambda without precise search
    """
    d = X.shape[1]
    Y = X.ravel("F")
    x = np.kron(np.eye(d), u.reshape(-1, 1))/w_2
    Y_ols = x @ la.inv(x.T @ x) @ x.T @ Y
    var_ols = ((Y-Y_ols)**2).mean()
    
    lambdas = np.exp(np.linspace(3, 3.5, 50))
    BICS = list(map(lambda lambd: BIC_new(x, Y, lambd, var_ols), lambdas))
    return lambdas[np.argmin(BICS)]

def ssvd_new(X, gamma_1 = 2, gamma_2 = 2, tol=1e-10, iters=10000):
    """
    vectorized version of sparse SVD algorithm
    """
    # shapes of X
    n = X.shape[0]
    d = X.shape[1]
    # find starting vectors
    U, S, V = la.svd(X)
    u_old = U[:, 0]
    v_old = V[0, :]
    for i in range(iters):
        # update v
        v_sim = np.zeros(d) ## v_sim initialization
        v_sh = X.T @ u_old ## v with sim and hat
        w_2 = np.vectorize(lambda x: 0 if x==0 else x**(-gamma_2))(abs(v_sh)) ## find w_2
        lambda_v = find_lambda_new(X, u_old, w_2)
        v_sim = sign_new(v_sh)*pos_new(abs(v_sh)-lambda_v*w_2/2)
        v_new = v_sim / la.norm(v_sim) ## normalization
        # update u
        u_sim = np.zeros(n) ## v_sim initialization
        u_sh = X @ v_new ## v with sim and hat
        w_1 = np.vectorize(lambda x: 0 if x==0 else x**(-gamma_1))(abs(u_sh)) ## find w_1
        lambda_u = find_lambda_new(X.T, v_new, w_1)
        u_sim = sign_new(u_sh)*pos_new(abs(u_sh)-lambda_u*w_1/2)
        u_new = u_sim / la.norm(u_sim) ## normalization
        # check convergence
        if la.norm(u_new - u_old)<tol and la.norm(v_new - v_old)<tol:
            break
        else:
            u_old = u_new
            v_old = v_new
    return u_new, v_new, u_new.T @ X @ v_new

def ssvd_approx(X, layer=1):
    """
    Given the number of layers, return the fitted matrix using sparse SVD algorithm as well as the F-distance in each step.
    """
    residual = X
    X_fit = np.zeros(X.shape)
    f_dist = []
    for i in range(layer):
        u_new, v_new, s = ssvd_new(residual)
        X_fit += s * u_new.reshape(-1, 1) @ v_new.reshape(1, -1)
        residual = X - X_fit
        f_dist.append(np.sqrt(((X - X_fit)**2).sum()))
    return X_fit, f_dist