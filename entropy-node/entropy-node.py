import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Return the Shannon entropy of the class labels.
    """
    # Write code here
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts = True)
    prob = counts/len(y)
    return -np.sum(prob * np.log2(prob)).astype(float)