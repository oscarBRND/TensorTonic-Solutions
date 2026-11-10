import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X = np.asarray(X, float)
    return np.linalg.inv(X.T @ X + lam*np.eye(len(X[0]))) @ X.T @ y 