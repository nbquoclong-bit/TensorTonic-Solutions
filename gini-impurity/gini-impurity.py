import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left, y_right = map(np.array, (y_left, y_right))
    N_l, N_r, N = len(y_left), len(y_right), len(y_left) + len(y_right)
    if N_l == 0 and N_r == 0:
        return 0.0
    val_l, count_l = np.unique(y_left, return_counts = True)
    G_left = 1.0 - np.sum((count_l/len(y_left)) ** 2)
    val_r, count_r = np.unique(y_right, return_counts = True)
    G_right = 1.0 - np.sum((count_r/len(y_right)) ** 2)
    
    
    return N_l/N * G_left + N_r/N * G_right