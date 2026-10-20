import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """
    n_v = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    n_w = np.sqrt(w[0]**2+w[1]**2+w[2]**2)
    dot_nw = v[0]*w[0] + v[1]*w[1] + v[2]*w[2]
    if (n_v == 0) or (n_w ==0):
        return np.nan
    cos = dot_nw/(n_v*n_w)
    theta = np.arccos(cos)
    return theta
    