import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    p_pred = [min(1-eps, max(eps, i)) for i in y_pred]
    L = [-(y_true[i] * math.log(p_pred[i]) + (1-y_true[i]) * math.log(1-p_pred[i])) for i in range(len(y_true))]
    return L