# These are the utilities used in the notebooks

import numpy as np

## Generating data
def generate_linear_1D(n_samples = 100, x_range = None, coeffs = None, intercept = None, noise = None, seed = None):
    if x_range is None:
        x_range = (0, 1,)
    if coeffs is None:
        coeffs = (1,)
    if intercept is None:
        intercept = 0
    if seed is not None:
        np.random.seed(seed)    
    X = np.linspace(x_range[0], x_range[1], n_samples)
    y = intercept + X * coeffs
    
    if noise is not None:
        y += np.random.normal(scale=noise, size=y.shape)
    
    # X, y have to be row vectors
    X = np.matrix(X).T
    y = np.matrix(y).T
    return X, y
    
def add_one_column(X_orig):
    X = np.matrix(X_orig)
    intercept_column = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept_column, X), axis=1)
    return X
    
## Define OLS solution
def ols(X_orig, y):
    X = add_one_column(X_orig)
    Xt = np.matrix(X).T
    XtX = np.dot(Xt, X)
    inv = np.linalg.inv(XtX)
    omega = np.dot(np.dot(inv,Xt), y)
    return np.array(omega.T)[0]

def ridge(X_orig, y, lamb):
    X = add_one_column(X_orig)
    Xt = np.matrix(X).T
    XtX = np.dot(Xt, X)
    Lamb = lamb*lamb # This is the only thing that is needed
    XtX += Lamb*np.identity(XtX.shape[0]) # Well, and this
    inv = np.linalg.inv(XtX)
    omega = np.dot(np.dot(inv,Xt), y)
    return np.array(omega.T)[0]

    