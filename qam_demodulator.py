import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

def qam_demodulator(symbols, M):
    """
    Demodulates QAM symbols back to original data symbols.
    
    Parameters:
    - symbols: QAM modulated complex symbols
    - M: QAM modulation order
    
    Returns:
    - Demodulated integer data symbols
    """
    # Decode symbols using angle and QAM constellation mapping
    demod_data = (np.angle(symbols) / (np.pi / M) - 1) / 2
    return np.round(demod_data).astype(int) % M