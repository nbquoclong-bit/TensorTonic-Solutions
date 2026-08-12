import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    z = []
    x = np.asarray(x)
    z = 1/(1+np.exp(-x))
    # Write code here
    return z