import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

# Channel Model Functions
def mcm_channel_model(u, initial_time, number_of_summations, symbol_duration, f_dmax, channel_coefficients):
    """
    Implements Monte Carlo Multipath (MCM) channel model simulation.
    
    Parameters:
    - u: Random variables for channel realization
    - initial_time: Starting time of the symbol
    - number_of_summations: Number of summation terms for channel modeling
    - symbol_duration: Duration of each symbol
    - f_dmax: Maximum Doppler frequency
    - channel_coefficients: Predefined channel tap gains
    
    Returns:
    - h: Channel impulse response
    - t_next: Next time step
    """
    t = initial_time
    Channel_Length = len(channel_coefficients)
    h_vector = []

    for k in range(Channel_Length):
        u_k = u[k, :]
        phi = 2 * np.pi * u_k
        f_d = f_dmax * np.sin(2 * np.pi * u_k)
        # Calculate channel tap using stochastic model
        h_tem = channel_coefficients[k] * 1 / np.sqrt(number_of_summations) * np.sum(np.exp(1j * phi) * np.exp(1j * 2 * np.pi * f_d * t))
        h_vector.append(h_tem)

    h = np.array(h_vector)
    t_next = initial_time + symbol_duration
    return h, t_next