import numpy as np

def time_based_split(timestamps: list, train_end: int, gap: int, val_end: int) -> dict:
    """
    Returns a dictionary with train_idx and val_idx as sorted integer NumPy arrays.
    """
    train_idx = []
    val_idx = []

    for i, elem in enumerate(timestamps):
        if elem < train_end:
            train_idx.append(i)
        elif (train_end + gap) <= elem < val_end:
            val_idx.append(i)
    return {
        "train_idx": np.asarray(train_idx),
        "val_idx": np.asarray(val_idx)
    }
