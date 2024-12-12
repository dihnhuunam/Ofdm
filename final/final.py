import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.linalg import toeplitz
from skimage.io import imread

# Simulation parameters
mod_method = 'QPSK'  # Modulation method
n_fft = 256         # IFFT/FFT size
n_cpe = 32          # Cyclic prefix extension size
snr = 30             # Target SNR (dB)
n_taps = 8           # Number of channel taps
ch_est_method = 'MMSE'  # Channel estimation method
save_file = False    # Option to save plot to file

# Modulation mapping
mod_methods = {'BPSK': 1, 'QPSK': 2, '8PSK': 3, '16QAM': 4, '32QAM': 5, '64QAM': 6}
mod_order = mod_methods[mod_method]

# Input data to binary stream
im = imread('logo.png')
im_bin = np.unpackbits(im.flatten())

# Binary stream to symbols
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=int)
im_bin_padded = np.concatenate((im_bin, padding))
cons_data = im_bin_padded.reshape(-1, mod_order)
# Fix cons_sym_id calculation
cons_sym_id = np.array([int("".join(map(str, bits)), 2) for bits in cons_data])

# Fix symbol_book initialization
if mod_order == 1:  # BPSK
    symbol_book = np.array([-1, 1])
elif mod_order in [2, 3]:  # PSK
    n = np.arange(2 ** mod_order)
    symbol_book = np.exp(1j * (2 * np.pi * n / (2 ** mod_order) + np.pi / 4))
elif mod_order == 5:  # 32QAM
    qam_dim = 6
    x = np.linspace(-qam_dim + 1, qam_dim - 1, qam_dim)
    symbol_book = (x[:, None] + 1j * x).flatten()
    mask = np.array([not ((i == 0 or i == qam_dim - 1) and (j == 0 or j == qam_dim - 1))
                     for i in range(qam_dim) for j in range(qam_dim)])
    symbol_book = symbol_book[mask]

X = symbol_book[cons_sym_id]

# IFFT to time domain
fft_rem = (n_fft - len(X) % n_fft) % n_fft
X_padded = np.concatenate((X, np.zeros(fft_rem, dtype=complex)))
X_blocks = X_padded.reshape(-1, n_fft)
x = np.fft.ifft(X_blocks, axis=1)

# Add cyclic prefix extension and serialize
x_cpe = np.hstack((x[:, -n_cpe:], x))
x_s = x_cpe.flatten()

# Add AWGN
signal_power = np.mean(np.abs(x_s) ** 2)
noise_power = signal_power / 10**(snr / 10)
noise = np.sqrt(noise_power / 2) * (np.random.randn(*x_s.shape) + 1j * np.random.randn(*x_s.shape))
x_s_noise = x_s + noise

# Fading channel
g = np.exp(-np.arange(n_taps))
g = g / np.linalg.norm(g)
x_s_noise_fading = np.convolve(x_s_noise, g, mode='same')

# Remove cyclic prefix and FFT
x_p = x_s_noise_fading.reshape(-1, n_fft + n_cpe)
x_p_cpr = x_p[:, n_cpe:]
X_hat_blocks = np.fft.fft(x_p_cpr, axis=1)

# # Channel estimation
# g_toeplitz = toeplitz(g, np.zeros(n_fft))
# if ch_est_method == 'LS':
#     G = X_hat_blocks[:, 0] / X_blocks[:, 0]
#     X_hat_blocks /= G[:, None]
# elif ch_est_method == 'MMSE':
#     H = np.fft.fft(g, n_fft)
#     H_conj = np.conj(H)
#     H_abs2 = np.abs(H) ** 2
#     W_mmse = H_conj / (H_abs2 + noise_power / signal_power)
#     X_hat_blocks *= W_mmse
# Cải thiện ước lượng kênh
if ch_est_method == 'MMSE':
    H = np.fft.fft(g, n_fft)
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    # Thêm hệ số regularization
    reg_factor = 0.01
    W_mmse = H_conj / (H_abs2 + reg_factor + noise_power / signal_power)
    X_hat_blocks *= W_mmse

# Thêm bộ lọc nhiễu
def apply_noise_filter(X_hat):
    # Simple moving average filter
    window = 3
    filtered_real = np.convolve(np.real(X_hat), np.ones(window)/window, mode='same')
    filtered_imag = np.convolve(np.imag(X_hat), np.ones(window)/window, mode='same')
    return filtered_real + 1j*filtered_imag

# Symbol demodulation
X_hat = X_hat_blocks.flatten()[:len(X)]
rec_syms = np.argmin(distance.cdist(np.column_stack((np.real(X_hat), np.imag(X_hat))),
                                    np.column_stack((np.real(symbol_book), np.imag(symbol_book)))), axis=1)
X_hat = apply_noise_filter(X_hat)

# Recover binary stream
rec_syms_bin = np.unpackbits(rec_syms.astype(np.uint8), axis=0)[:len(im_bin)]
ber = np.mean(rec_syms_bin != im_bin)

# Recover image
rec_im = np.packbits(rec_syms_bin).reshape(im.shape)

# Generate plots
plt.figure(figsize=(10, 8))

# Transmit constellation
plt.subplot(2, 2, 1)
plt.plot(np.real(X), np.imag(X), 'x')
plt.title(f'Transmitted Constellation\n{mod_method} Modulation')
plt.grid(True)

# Received constellation
plt.subplot(2, 2, 2)
plt.plot(np.real(X_hat), np.imag(X_hat), 'x')
plt.title(f'Received Constellation\nMeasured SNR: {snr:.2f} dB')
plt.grid(True)

# Original image
plt.subplot(2, 2, 3)
plt.imshow(im, cmap='gray')
plt.title('Original Image')

# Recovered image
plt.subplot(2, 2, 4)
plt.imshow(rec_im, cmap='gray')
plt.title(f'Recovered Image\nBER: {ber:.2g}')

plt.tight_layout()
plt.savefig('Result3.png')
plt.show()

