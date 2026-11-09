import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    n = len(y_pred)
    MSE = 0
    for i in range(n):
        MSE += (y_pred[i]-y_true[i])**2
    return MSE/n