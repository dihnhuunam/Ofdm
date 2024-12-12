import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import convolve
from skimage.io import imread

# Parameters
mod_method = 'QPSK'
n_fft = 64
n_cpe = 16
snr = 30  # in dB
n_taps = 4
ch_est_method = 'MMSE'

# Modulation configuration
mod_methods = ['BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM']
mod_order = mod_methods.index(mod_method) + 1

# Load image
im = imread('VN.png')

if im.shape[-1] == 4:
    im = im[:, :, :3]

im_bin = []
for i in range(3):  # Xử lý từng kênh màu
    channel = (im[:, :, i] * 255).astype(np.uint8) if im.dtype != np.uint8 else im[:, :, i]
    im_bin.append(np.unpackbits(channel.flatten()))

im_bin = np.concatenate(im_bin)  # Gộp các kênh lại

# # Padding and symbol mapping
# sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
# padding = np.zeros(sym_rem, dtype=np.uint8)
# im_bin_padded = np.concatenate((im_bin, padding))
# cons_data = im_bin_padded.reshape(-1, mod_order)
# cons_sym_id = np.packbits(cons_data, axis=1).flatten()

# Generate modulation symbol book
if mod_method in ['BPSK', 'QPSK', '8PSK']:
    mod_ind = 2**mod_order
    angles = np.linspace(0, 2 * np.pi, mod_ind, endpoint=False)
    symbol_book = np.exp(1j * angles)
elif mod_method in ['16QAM', '64QAM', '32QAM']:
    mod_ind = int(np.sqrt(2**mod_order))
    in_phase = np.linspace(-1, 1, mod_ind)
    quadrature = np.linspace(-1, 1, mod_ind)
    in_phase, quadrature = np.meshgrid(in_phase, quadrature)
    symbol_book = in_phase.flatten() + 1j * quadrature.flatten()
    if mod_method == '32QAM':
        mask = np.abs(in_phase.flatten()) + np.abs(quadrature.flatten()) > 1
        symbol_book = symbol_book[~mask]

# Padding và ánh xạ ký hiệu
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=np.uint8)
im_bin_padded = np.concatenate((im_bin, padding))
cons_data = im_bin_padded.reshape(-1, mod_order)
cons_sym_id = np.dot(cons_data, 1 << np.arange(cons_data.shape[1])[::-1])
cons_sym_id = cons_sym_id % len(symbol_book)  # Đảm bảo trong phạm vi hợp lệ

# Chuyển đổi từ bit nhị phân sang chỉ số ký hiệu
cons_sym_id = np.dot(cons_data, 1 << np.arange(cons_data.shape[1])[::-1])
cons_sym_id = cons_sym_id % len(symbol_book)  # Giới hạn chỉ số trong phạm vi hợp lệ

# Kiểm tra tính hợp lệ
assert np.all(cons_sym_id < len(symbol_book)), "cons_sym_id vẫn chứa giá trị ngoài phạm vi!"

# Điều chế dữ liệu
X = symbol_book[cons_sym_id]

# OFDM block creation
fft_rem = (n_fft - len(X) % n_fft) % n_fft
X_padded = np.concatenate((X, np.zeros(fft_rem, dtype=np.complex64)))
X_blocks = X_padded.reshape(-1, n_fft)
x = np.fft.ifft(X_blocks, axis=1)

# Add cyclic prefix extension
x_cpe = np.hstack((x[:, -n_cpe:], x))
x_s = x_cpe.flatten()

# Add noise
data_pwr = np.mean(np.abs(x_s)**2)
noise_pwr = data_pwr / (10**(snr / 10))
noise = np.sqrt(noise_pwr / 2) * (np.random.randn(*x_s.shape) + 1j * np.random.randn(*x_s.shape))
x_s_noise = x_s + noise

# Fading channel
g = np.exp(-np.arange(n_taps))
g /= np.linalg.norm(g)
x_s_noise_fading = convolve(x_s_noise, g, mode='same')

# Reshape back to OFDM blocks
x_p = x_s_noise_fading.reshape(-1, n_fft + n_cpe)
x_p_cpr = x_p[:, n_cpe:]

# Move to frequency domain
X_hat_blocks = np.fft.fft(x_p_cpr, axis=1)

# Channel estimation and equalization
if n_taps > 1:
    if ch_est_method == 'MMSE':
        G = X_hat_blocks[:, 0] / X_blocks[:, 0]
        G_mmse = np.conj(G) / (np.abs(G)**2 + (noise_pwr / (data_pwr * 0.5)))  # Tỷ lệ SNR
        X_hat_blocks *= G_mmse[:, None]  # Áp dụng cho toàn bộ tín hiệu

X_hat = X_hat_blocks.flatten()[:len(X)]

# Demodulation
rec_syms = np.array([np.argmin(np.abs(symbol_book - sym)) for sym in X_hat])

# Recover binary data
rec_syms_cons = np.unpackbits(rec_syms.astype(np.uint8).reshape(-1, 1), axis=1)[:, -mod_order:]
rec_im_bin = rec_syms_cons.flatten()[:len(im_bin)]
ber = np.sum(rec_im_bin != im_bin) / len(im_bin)

# Recover image
# Phục hồi từng kênh từ dữ liệu nhị phân
rec_im_bin_split = np.split(rec_im_bin, 3)
rec_channels = []
for i in range(3):
    rec_channels.append(np.packbits(rec_im_bin_split[i]).reshape(im[:, :, i].shape))

# Gộp lại thành ảnh màu
rec_im = np.stack(rec_channels, axis=-1)

# Visualization
plt.figure(figsize=(10, 10))

# Transmit constellation
plt.subplot(2, 2, 1)
plt.plot(X.real, X.imag, 'x', linewidth=2, markersize=10)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('In phase')
plt.ylabel('Quadrature')
plt.title(f'Transmit Constellation\n{mod_method} Modulation')
plt.grid()

# Received constellation
plt.subplot(2, 2, 2)
plt.plot(X_hat.real[::500], X_hat.imag[::500], 'x', markersize=3)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('In phase')
plt.ylabel('Quadrature')
plt.title(f'Received Constellation\nMeasured SNR: {snr:.2f} dB')
plt.grid()

# Original image
plt.subplot(2, 2, 3)
plt.imshow(im, cmap='gray')
plt.title('Transmit Image')

# Recovered image
plt.subplot(2, 2, 4)
plt.imshow(rec_im, cmap='gray')
plt.title(f'Recovered Image\nBER: {ber:.2g}')

plt.tight_layout()
plt.savefig('rusult3.png')
plt.show()
