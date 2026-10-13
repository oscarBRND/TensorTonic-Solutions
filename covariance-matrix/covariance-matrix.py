import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """

    X = np.asarray(X)
    means = np.mean(X, axis=0)
    X_c = X - means

    cov = (np.transpose(X_c) @ X_c)/(len(X)-1)
    return cov