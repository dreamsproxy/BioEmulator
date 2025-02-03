import numpy as np
from numba import jit

@jit(nopython=True)
def tanh(x):
    """
    Numba JIT-compiled implementation of the TanH activation function.
    
    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array with TanH applied element-wise.
    """
    result = np.empty_like(x)
    for i in range(x.size):
        result[i] = np.tanh(x.flat[i])  # Efficiently handle arrays of any shape
    return result

@jit(nopython=True)
def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    """
    return np.maximum(alpha * x, x)

@jit(nopython=True)
def softmax(x, boost):
    """
    Softmax Activation function
    """
    e_x = np.exp(x - np.max(x))
    e_x /= e_x.sum()
    return e_x * boost