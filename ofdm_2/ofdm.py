import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64                  # Number of subcarriers
CP = 16                 # Cyclic Prefix Length
M = 16                  # 16-QAM Modulation
num_symbols = 1000      # Number of OFDM symbols
SNR_dB = 20             # Signal-to-Noise Ratio in dB

# Helper Functions
def qam_mod(data, M):
    """16-QAM Modulation"""
    m = int(np.sqrt(M))
    real = 2 * (data % m) - m + 1
    imag = 2 * (data // m) - m + 1
    return real + 1j * imag

def qam_demod(data, M):
    """16-QAM Demodulation"""
    m = int(np.sqrt(M))
    real = np.round((data.real + m - 1) / 2).astype(int) % m
    imag = np.round((data.imag + m - 1) / 2).astype(int) % m
    return imag * m + real

def add_awgn_noise(signal, snr_dB):
    """Add AWGN noise to signal"""
    snr_linear = 10 ** (snr_dB / 10)
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# Transmitter
data = np.random.randint(0, M, (num_symbols, N))  # Random bits
mod_data = qam_mod(data, M)                       # 16-QAM Modulation
ifft_data = np.fft.ifft(mod_data, axis=1)         # IFFT
tx_signal = np.hstack([ifft_data[:, -CP:], ifft_data])  # Add Cyclic Prefix

# Channel: Rayleigh fading + AWGN
h = (1 / np.sqrt(2)) * (np.random.randn(num_symbols, N) + 1j * np.random.randn(num_symbols, N))  # Rayleigh fading
rx_signal = np.zeros_like(tx_signal, dtype=complex)
for i in range(num_symbols):
    rx_signal[i] = np.convolve(tx_signal[i], h[i], mode='same')  # Convolution with channel

rx_signal = add_awgn_noise(rx_signal, SNR_dB)  # Add AWGN noise

# Receiver
rx_signal_no_cp = rx_signal[:, CP:]            # Remove Cyclic Prefix
fft_data = np.fft.fft(rx_signal_no_cp, axis=1)  # FFT
h_est = h                                      # Perfect channel estimation for simplicity
mmse_eq = np.conj(h_est) / (np.abs(h_est) ** 2 + (1 / (10 ** (SNR_dB / 10))))  # MMSE Equalizer
equalized_data = fft_data * mmse_eq            # Equalization
demod_data = qam_demod(equalized_data, M)      # 16-QAM Demodulation

# BER Calculation
bit_errors = np.sum(demod_data != data)
ber = bit_errors / (N * num_symbols)

# Results
print(f"Bit Error Rate (BER): {ber}")

# Plotting
plt.figure(figsize=(10, 6))
plt.title("Constellation Diagram (16-QAM)")
plt.scatter(equalized_data.real.flatten(), equalized_data.imag.flatten(), s=2, alpha=0.5)
plt.grid(True)
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.axis([-5, 5, -5, 5])
plt.show()
