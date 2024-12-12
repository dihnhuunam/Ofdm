import numpy as np

def symerr(a, b):
    """
    Calculates symbol error rate (SER) and number of errors.
    
    Parameters:
    - a: Original data symbols
    - b: Decoded data symbols
    
    Returns:
    - Number of symbol errors
    - Symbol error rate
    """
    num_errors = np.sum(a != b)
    error_rate = num_errors / len(a)
    return num_errors, error_rate