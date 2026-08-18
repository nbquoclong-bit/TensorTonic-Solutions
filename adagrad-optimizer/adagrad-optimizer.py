import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    G_new = np.array(G) + np.array(g) ** 2
    w_new = np.array(w) - lr/np.sqrt(np.array(G_new) + eps) * (np.array(g))
    return w_new, G_new