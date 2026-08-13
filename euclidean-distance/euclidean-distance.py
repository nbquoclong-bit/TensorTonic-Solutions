import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x_vec = np.array(x)
    y_vec = np.array(y)
    dist = 0.0
    if x_vec.shape[0] != y_vec.shape[0]:
        raise ValueError
    for i in range(x_vec.shape[0]):
        dist += (x_vec[i] - y_vec[i]) ** 2
    return np.sqrt(dist)