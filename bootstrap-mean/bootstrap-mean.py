import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """

    rng = np.random.default_rng(seed)

    x = np.asarray(x)
    indices = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    samples = x[indices]
    
    means = samples.mean(axis=1)
    
    alpha = (1-ci)/2
    q = np.quantile(means, [alpha, (1-alpha)])
    
    mean = means.mean()
    
    return {
        "bootstrap_mean": float(mean),
        "lower": float(q[0]),
        "upper": float(q[1])
    }