import numpy as np

def awgn(s, snr_dB):
    """
    Adds Additive White Gaussian Noise (AWGN) to the signal.
    
    Parameters:
    - s: Input signal
    - snr_dB: Signal-to-Noise Ratio in decibels
    
    Returns:
    - Noisy signal with AWGN added
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_dB / 10)
    # Calculate signal power
    power_signal = np.mean(np.abs(s)**2)
    # Calculate noise variance
    noise_variance = power_signal / snr_linear
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape))
    return s + noise
