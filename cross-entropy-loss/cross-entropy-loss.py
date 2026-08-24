import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Return the mean multiclass cross-entropy loss.
    """
    y_true, y_pred = [np.array(x) for x in (y_true, y_pred)]
    L_arr = np.zeros(len(y_true), dtype = 'float')
    for i in range(len(y_true)):
        L_arr[i] = -np.log(y_pred[i][y_true[i]])
    L = np.mean(L_arr)
    return L