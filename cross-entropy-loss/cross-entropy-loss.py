import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    sum_loss = np.zeros(y_t.shape[0])
    for i in np.arange(y_t.shape[0]):
        sum_loss[i] = np.log(y_p[i][y_t[i]])
    return -np.mean(sum_loss)