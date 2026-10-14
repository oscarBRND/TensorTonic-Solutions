import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """Returns the correlation matrix as a NumPy array."""
    X = np.asarray(X, dtype=float)

    cov = np.cov(X, rowvar=False)          # variables en colonnes
    std = np.std(X, axis=0, ddof=1)        # même ddof que np.cov

    return cov / np.outer(std, std)        # division terme à terme