import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import convolve
from skimage.io import imread
from scipy.interpolate import interp1d

# User input for modulation method
mod_method = input("Choose Modulation (BPSK, QPSK, 8PSK, 16QAM, 32QAM, 64QAM): ")

# Simulation Parameters
n_fft = 128  # FFT size
n_cpe = 16   # Cyclic prefix length
snr = 20     # Signal-to-Noise Ratio in dB
n_taps = 4   # Number of channel taps
ch_est_method = 'MMSE'  # Channel estimation method

# Error correction coding parameters
code_rate = 2/3
n_bits_per_block = 48

# Functions for error correction coding
def hamming_encode(data):
    """Perform simple Hamming encoding with parity check."""
    n = len(data)
    pad_len = (-n) % n_bits_per_block  # Pad data to match block size
    if pad_len:
        data = np.pad(data, (0, pad_len))
    
    encoded = []
    for i in range(0, len(data), n_bits_per_block):
        block = data[i:i + n_bits_per_block]
        parity = np.sum(block) % 2  # Calculate parity bit
        encoded.extend(block)
        encoded.append(parity)
    
    return np.array(encoded)

def hamming_decode(data):
    """Decode Hamming-coded data with error correction."""
    decoded = []
    for i in range(0, len(data), n_bits_per_block + 1):
        block = data[i:i + n_bits_per_block]
        parity = data[i + n_bits_per_block]
        # Correct error using parity if detected
        if np.sum(block) % 2 != parity:
            error_pos = np.argmax(np.abs(np.diff(block)))
            block[error_pos] = 1 - block[error_pos]
        decoded.extend(block)
    return np.array(decoded)

# Map modulation method to modulation order
mod_methods = ['BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM']
mod_order = mod_methods.index(mod_method) + 1

# Load and preprocess image
im = imread('./images/image.png')
if im.shape[-1] == 4:  # Handle alpha channel
    im = im[:, :, :3]

# Convert image to binary
im_bin = []
for i in range(3):  # Process RGB channels
    channel = (im[:, :, i] * 255).astype(np.uint8) if im.dtype != np.uint8 else im[:, :, i]
    im_bin.append(np.unpackbits(channel.flatten()))
im_bin = np.concatenate(im_bin)

# Apply Hamming encoding
im_bin_encoded = hamming_encode(im_bin)

# Define symbol book based on modulation order
if mod_order == 1:  # BPSK
    symbol_book = np.array([-1, 1])
elif mod_order in [2, 3]:  # QPSK, 8PSK
    mod_ind = 2 ** (mod_order - 1)
    angles = np.linspace(0, 2 * np.pi, mod_ind, endpoint=False) + np.pi / 4
    symbol_book = np.cos(angles) + 1j * np.sin(angles)
elif mod_order in [4, 6]:  # 16QAM, 64QAM
    mod_ind = int(np.sqrt(2 ** mod_order))
    real_part = np.linspace(-1, 1, mod_ind)
    imag_part = np.linspace(-1, 1, mod_ind)
    symbol_book = np.array([x + 1j * y for x in real_part for y in imag_part])
elif mod_order == 5:  # 32QAM
    mod_ind = 6
    real_part = np.linspace(-1, 1, mod_ind)
    imag_part = np.linspace(-1, 1, mod_ind)
    symbol_book = np.array([x + 1j * y for x in real_part for y in imag_part])
    corners = [0, 5, 30, 35]  # Remove corner points
    symbol_book = np.delete(symbol_book, corners)

# Symbol mapping with Gray coding
sym_rem = (mod_order - len(im_bin_encoded) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=np.uint8)
im_bin_padded = np.concatenate((im_bin_encoded, padding))
cons_data = im_bin_padded.reshape(-1, mod_order)
cons_sym_id = np.dot(cons_data, 1 << np.arange(cons_data.shape[1])[::-1])
cons_sym_id = (cons_sym_id % len(symbol_book)).astype(int)

# Map bits to symbols
X = symbol_book[cons_sym_id]

# OFDM setup with pilot structure
n_pilots = n_fft // 8
pilot_indices = np.linspace(0, n_fft - 1, n_pilots, dtype=int)
pilot_values = np.exp(1j * 2 * np.pi * np.random.rand(n_pilots))

# Pad and reshape symbols for OFDM
fft_rem = (n_fft - len(X) % n_fft) % n_fft
X_padded = np.concatenate((X, np.zeros(fft_rem, dtype=np.complex64)))
X_blocks = X_padded.reshape(-1, n_fft)

# Insert pilots
for i, idx in enumerate(pilot_indices):
    X_blocks[:, idx] = pilot_values[i]

# Perform IFFT and add cyclic prefix
x = np.fft.ifft(X_blocks, axis=1)
x_cpe = np.hstack((x[:, -n_cpe:], x))
x_s = x_cpe.flatten()

# Add noise
data_pwr = np.mean(np.abs(x_s) ** 2)
noise_pwr = data_pwr / (10 ** (snr / 10))
noise = np.sqrt(noise_pwr / 2) * (np.random.randn(*x_s.shape) + 1j * np.random.randn(*x_s.shape))
x_s_noise = x_s + noise

# Apply channel model
g = np.exp(-np.arange(n_taps) / 2)  # Channel impulse response
g /= np.linalg.norm(g)
x_s_noise_fading = convolve(x_s_noise, g, mode='same')

# Receiver processing
x_p = x_s_noise_fading.reshape(-1, n_fft + n_cpe)
x_p_cpr = x_p[:, n_cpe:]
X_hat_blocks = np.fft.fft(x_p_cpr, axis=1)

# Channel estimation and equalization
if n_taps > 1 and ch_est_method == 'MMSE':
    H_pilots = X_hat_blocks[:, pilot_indices] / pilot_values
    H_interp = np.zeros_like(X_hat_blocks, dtype=np.complex128)
    for i in range(X_hat_blocks.shape[0]):
        interp_real = interp1d(pilot_indices, H_pilots[i].real, kind='cubic')
        interp_imag = interp1d(pilot_indices, H_pilots[i].imag, kind='cubic')
        H_interp[i] = interp_real(np.arange(n_fft)) + 1j * interp_imag(np.arange(n_fft))
    noise_var = noise_pwr
    H_mmse = np.conj(H_interp) / (np.abs(H_interp) ** 2 + noise_var / data_pwr)
    X_hat_blocks *= H_mmse

# Flatten received blocks and demodulate
X_hat = X_hat_blocks.flatten()[:len(X)]
tree = cKDTree(np.column_stack((symbol_book.real, symbol_book.imag)))
_, rec_syms = tree.query(np.column_stack((X_hat.real, X_hat.imag)))

# Convert symbols to bits
rec_syms_cons = np.unpackbits(rec_syms.astype(np.uint8).reshape(-1, 1), axis=1)[:, -mod_order:]
rec_im_bin_encoded = rec_syms_cons.flatten()[:len(im_bin_encoded)]

# Decode received bits and calculate BER
rec_im_bin = hamming_decode(rec_im_bin_encoded)[:len(im_bin)]
ber = np.sum(rec_im_bin != im_bin) / len(im_bin)

# Recover the image
rec_im_bin_split = np.split(rec_im_bin, 3)
rec_channels = []
for i in range(3):
    rec_channels.append(np.packbits(rec_im_bin_split[i]).reshape(im[:, :, i].shape))
rec_im = np.stack(rec_channels, axis=-1)

# Visualization
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(X.real, X.imag, 'x', linewidth=2, markersize=10)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('In phase')
plt.ylabel('Quadrature')
plt.title(f'Transmit Constellation\n{mod_method} Modulation')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(X_hat.real[::500], X_hat.imag[::500], 'x', markersize=3)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('In phase')
plt.ylabel('Quadrature')
plt.title(f'Received Constellation\nMeasured SNR: {snr:.2f} dB')
plt.grid()

plt.subplot(2, 2, 3)
plt.imshow(im)
plt.title('Transmit Image')

plt.subplot(2, 2, 4)
plt.imshow(rec_im)
plt.title(f'Recovered Image\nBER: {ber:.2g}')

plt.tight_layout()
plt.savefig('./results/result_improved.png')
plt.show()
