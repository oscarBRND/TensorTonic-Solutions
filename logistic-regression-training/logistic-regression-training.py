import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    N = len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        
        p = _sigmoid(X@w + b)
        dw = (1/N)*X.T@(p-y)
        db = (1/N)*np.sum(p-y)
        w = w - lr*dw
        b = b - lr*db
    return w, b 
        