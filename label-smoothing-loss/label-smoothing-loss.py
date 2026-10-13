import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    smoothed_target = []
    loss = 0
    for i in range(len(predictions)):
        if i == target:
            smoothed_target.append((1-epsilon) + epsilon/len(predictions))
        else:
            smoothed_target.append(epsilon/len(predictions))

        loss += smoothed_target[-1] * math.log(predictions[i])

    return -loss

    