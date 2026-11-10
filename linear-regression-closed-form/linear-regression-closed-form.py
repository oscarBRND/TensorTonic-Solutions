import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X = np.asarray(X, float)
    result = ((np.linalg.inv(X.T @ X)) @ X.T) @ y     
    return result 
    