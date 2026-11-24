import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    grad, m, v, w = (np.asarray(x, dtype=np.float64) for x in (grad, m, v, w))
    
    new_m = beta1*m + (1-beta1)*grad
    new_v = beta2*v + (1-beta2)*(grad**2)
    new_w = w - lr*((beta1*new_m + (1-beta1)*grad)/(np.sqrt(new_v) + eps))
    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
    