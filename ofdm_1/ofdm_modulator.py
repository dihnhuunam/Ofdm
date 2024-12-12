import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

def ofdm_modulator(data, NFFT, G):
    """
    Modulates input data into an OFDM signal.
    
    Parameters:
    - data: Input data symbols
    - NFFT: Fast Fourier Transform size
    - G: Guard interval size
    
    Returns:
    - OFDM modulated time-domain signal
    """
    chnr = len(data)
    # Zero-pad the data to NFFT size
    x = np.concatenate([data, np.zeros(NFFT - chnr)])
    # Convert to time domain using IFFT
    a = ifft(x)
    # Add cyclic prefix (guard interval)
    y = np.concatenate([a[-G:], a])
    return y