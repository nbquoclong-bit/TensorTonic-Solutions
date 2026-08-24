import numpy as np

def softmax(x: list) -> np.ndarray:
    """Return stable softmax probabilities with the same shape as x."""
    # Write code here
    x = np.array(x)
    if x.ndim == 1:
        p = np.zeros((1, len(x)))
        x = x - np.max(x)
        sum = np.sum(np.exp(x))
        p = np.exp(x)/sum
    else:
        p = np.zeros((len(x), len(x[0])))
        for i in range(len(x)):
            x[i] = x[i] - np.max(x[i])
            sum = np.sum(np.exp(x[i]))
            p[i] = np.exp(x[i])/sum
    return p