import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m_new = beta1 * np.array(m) + (1-beta1)*np.array(grad)
    v_new = beta2 * np.array(v) + (1-beta2)*(np.array(grad)**2)
    m_diff = m_new/(1-beta1 ** t)
    v_diff = v_new/(1-beta2 ** t)
    param_new = param - lr * (m_diff/(np.sqrt(v_diff) + eps))
    
    return param_new, m_new, v_new