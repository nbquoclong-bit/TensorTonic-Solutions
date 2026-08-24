import numpy as np

def softmax(x: list) -> np.ndarray:
    """Return stable softmax probabilities with the same shape as x."""
    # Write code here
    x = np.array(x, dtype = float)
    if x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x)/np.sum(np.exp(x))
    x = x - np.max(x, axis = 1, keepdims = True)
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)