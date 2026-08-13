import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.array(A)
    n_sam, n_feat = A.shape
    trace = 0
    if n_sam == n_feat:
        for i in range(n_sam):
            trace += A[i][i]
    return trace
