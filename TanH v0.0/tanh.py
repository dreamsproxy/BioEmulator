import numpy as np
from numba import jit

@jit(nopython=True)
def tanh_activation(x):
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
