import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    z = 0.0
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    return float(sum(np.abs(x_array - y_array)))