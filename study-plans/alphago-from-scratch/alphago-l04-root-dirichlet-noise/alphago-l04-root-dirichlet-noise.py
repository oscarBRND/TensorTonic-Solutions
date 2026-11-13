import numpy as np

def mix_root_dirichlet_noise(
    priors: np.ndarray,
    noise: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Returns the mixed float64 prior array in the original action order.
    """
    sum_prior = 0
    sum_noise = 0
    for i, elem in enumerate(legal_mask):
        if elem:
            sum_prior+=priors[i]
            sum_noise+=noise[i]

    result = []
    for i, elem in enumerate(legal_mask):
        if elem:
            result.append((1-epsilon)*(priors[i]/sum_prior) + epsilon*(noise[i]/sum_noise))
        else:
            result.append(0)

    return np.asarray(result, float)