import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    new_m = beta1 * np.asarray(m) + (1-beta1) * np.asarray(grad)
    
    new_v = beta2 * np.asarray(v) + (1-beta2) * (np.asarray(grad)**2)

    m_tilda = beta1 * np.asarray(new_m) + (1-beta1) * np.asarray(grad)
    new_w = np.asarray(w) - lr*(m_tilda/(np.sqrt(new_v) + eps))
    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
    