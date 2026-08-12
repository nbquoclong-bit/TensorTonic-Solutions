import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    x = float(np.dot(np.array(a), np.array(b)))
    y = float(np.linalg.norm(np.array(a)))
    z = float(np.linalg.norm(np.array(b)))
    t = 0.0 if y == 0.0 or z == 0.0 else x/(y*z)
    # Write code here
    return t