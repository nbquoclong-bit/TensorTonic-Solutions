import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
        det_mat = np.linalg.det(matrix)
    except:
        return None
    eigvals, eigvec = np.linalg.eig(matrix)
    return eigvals