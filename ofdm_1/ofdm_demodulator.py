import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

def ofdm_demodulator(data, chnr, NFFT, G):
    """
    Demodulates OFDM signal back to data symbols.
    
    Parameters:
    - data: OFDM modulated signal
    - chnr: Number of data symbols
    - NFFT: FFT size
    - G: Guard interval size
    
    Returns:
    - Demodulated data symbols in frequency domain
    """
    # Remove guard interval
    x_remove_guard_interval = data[G:NFFT + G]
    # Convert back to frequency domain
    x = fft(x_remove_guard_interval)
    # Extract original data symbols
    y = x[:chnr]
    return y