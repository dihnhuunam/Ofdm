import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import convolve
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

# Simulation parameters
mod_method = 'QPSK'
n_fft = 64
n_cpe = 16
snr = 20
n_taps = 8
ch_est_method = 'LS'

# Modulation methods
mod_methods = {'BPSK': 1, 'QPSK': 2, '8PSK': 3, '16QAM': 4, '32QAM': 5, '64QAM': 6}
mod_order = mod_methods[mod_method]

# Load and process image
im = imread('image.jpg', as_gray=True)  # Read image as grayscale
im = img_as_ubyte(im)
im_bin = ''.join(format(byte, '08b') for byte in im.ravel())

# Convert binary stream to symbols
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
im_bin_padded = im_bin + '0' * sym_rem
if mod_order == 1:  # Special case for BPSK
    symbols = [int(bit) for bit in im_bin_padded]
else:
    symbols = [int(im_bin_padded[i:i+mod_order], 2) for i in range(0, len(im_bin_padded), mod_order)]
    
# # Define symbol book
# if mod_order == 1:  # BPSK
#     symbol_book = np.array([-1, 1])
# elif mod_order in [2, 3]:  # QPSK, 8PSK
#     mod_ind = 2 ** (mod_order - 1)
#     angles = np.linspace(0, 2 * np.pi, mod_ind, endpoint=False) + np.pi / 4
#     symbol_book = np.cos(angles) + 1j * np.sin(angles)
# elif mod_order in [4, 6]:  # 16QAM, 64QAM
#     mod_ind = int(np.sqrt(2 ** mod_order))
#     real_part = np.linspace(-1, 1, mod_ind)
#     imag_part = np.linspace(-1, 1, mod_ind)
#     symbol_book = np.array([x + 1j * y for x in real_part for y in imag_part])
# elif mod_order == 5:  # 32QAM
#     mod_ind = 6
#     real_part = np.linspace(-1, 1, mod_ind)
#     imag_part = np.linspace(-1, 1, mod_ind)
#     symbol_book = np.array([x + 1j * y for x in real_part for y in imag_part])
#     corners = [0, 5, 30, 35]
#     symbol_book = np.delete(symbol_book, corners)

# # Modulate data
# X = np.array([symbol_book[s] for s in symbols])

# Define symbol book for QPSK
if mod_order == 2:  # QPSK
    mod_ind = 2 ** mod_order  # 4 symbols
    angles = np.linspace(0, 2 * np.pi, mod_ind, endpoint=False) + np.pi / 4
    symbol_book = np.cos(angles) + 1j * np.sin(angles)

# Verify symbols and symbol book match
assert max(symbols) < len(symbol_book), "Symbol index out of range for symbol book"

# Modulate data
X = np.array([symbol_book[s] for s in symbols])

# IFFT and add cyclic prefix
fft_rem = (n_fft - len(X) % n_fft) % n_fft
X_padded = np.concatenate([X, np.zeros(fft_rem)])
X_blocks = X_padded.reshape(-1, n_fft)
x = np.fft.ifft(X_blocks, axis=1)
x_cpe = np.hstack([x[:, -n_cpe:], x])
x_s = x_cpe.ravel()

# Add noise
data_pwr = np.mean(np.abs(x_s)**2)
noise_pwr = data_pwr / 10**(snr / 10)
noise = np.sqrt(noise_pwr / 2) * (np.random.randn(*x_s.shape) + 1j * np.random.randn(*x_s.shape))
x_s_noise = x_s + noise

# Apply fading channel
g = np.exp(-np.arange(n_taps))
g /= np.linalg.norm(g)
x_s_noise_fading = convolve(x_s_noise, g, mode='same')

# FFT and remove cyclic prefix
x_p = x_s_noise_fading.reshape(-1, n_fft + n_cpe)
x_p_cpr = x_p[:, n_cpe:]
X_hat_blocks = np.fft.fft(x_p_cpr, axis=1)

# Channel estimation
if n_taps > 1 and ch_est_method == 'LS':
    G = X_hat_blocks[:, 0] / X_blocks[:, 0]
    X_hat_blocks /= G[:, None]

# Demodulation
X_hat = X_hat_blocks.ravel()[:len(X)]
rec_syms = np.array([np.argmin(distance.cdist([[z.real, z.imag]], np.c_[symbol_book.real, symbol_book.imag])) for z in X_hat])

# Convert symbols back to binary
rec_bin = ''.join(format(s, f'0{mod_order}b') for s in rec_syms)
rec_bin = rec_bin[:len(im_bin)]

# Calculate BER
bit_errors = sum(a != b for a, b in zip(im_bin, rec_bin))
ber = bit_errors / len(im_bin)

# Recover image
rec_im = np.array([int(rec_bin[i:i+8], 2) for i in range(0, len(rec_bin), 8)], dtype=np.uint8).reshape(im.shape)

# Plot results
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(X.real, X.imag, s=10, label="Transmit Constellation")
plt.title(f"Transmit Constellation ({mod_method})")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid()

plt.subplot(2, 2, 2)
plt.scatter(X_hat.real, X_hat.imag, s=10, label="Received Constellation")
plt.title(f"Received Constellation (Measured SNR: {snr:.2f} dB)")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid()

plt.subplot(2, 2, 3)
plt.imshow(im, cmap='gray')
plt.title("Original Image")

plt.subplot(2, 2, 4)
plt.imshow(rec_im, cmap='gray')
plt.title(f"Recovered Image (BER: {ber:.2g})")

plt.tight_layout()
plt.savefig('Result.png')
plt.show()