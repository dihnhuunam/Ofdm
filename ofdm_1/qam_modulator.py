import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

def qam_modulator(data, M):
    """
    Modulates data symbols using Quadrature Amplitude Modulation (QAM).
    
    Parameters:
    - data: Input data symbols
    - M: QAM modulation order (constellation size)
    
    Returns:
    - QAM modulated complex symbols
    """
    return np.sqrt(1 / 10) * np.exp(1j * (np.pi / M) * (2 * data + 1))