import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true, y_pred = [np.array(x, dtype='float') for x in (y_true, y_pred)]
    e = y_true - y_pred
    L = float(np.mean(np.where(np.abs(e) <= delta, 1/2 * e**2, delta * (np.abs(e) - 1/2 * delta))))
    return L