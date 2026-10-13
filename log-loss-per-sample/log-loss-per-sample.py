import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """

    p_clip = [min(1-eps, max(eps,p)) for p in y_pred]
    
    log_loss = []
    for y, p_hat in zip(y_true, p_clip):
        log_loss.append(-(y*math.log(p_hat)+(1-y)*math.log(1-p_hat)))

    return log_loss