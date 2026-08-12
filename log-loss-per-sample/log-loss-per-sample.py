import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    true_y = np.asarray(y_true)
    pred_y = np.clip(np.asarray(y_pred), eps, 1-eps)
    loss = (-(true_y*np.log(pred_y)+(1-true_y)*np.log(1-pred_y))).tolist()
    return loss