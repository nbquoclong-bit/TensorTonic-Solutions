import numpy as np

def minmax_scale(X, axis, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X_scale = (X - np.min(X, axis, keepdims=True)) / np.maximum((np.max(X, axis, keepdims=True) - np.min(X, axis,keepdims=True)), eps)
    return X_scale