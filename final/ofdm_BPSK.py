import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage.io import imread

# Parameters
mod_method = 'BPSK'
n_fft = 128
n_cpe = 16
snr = 30  # in dB
n_taps = 8
ch_est_method = 'MMSE'

# Modulation configuration
mod_order = 1  # BPSK chỉ sử dụng 1 bit trên mỗi ký hiệu
symbol_book = np.array([-1, 1], dtype=np.complex64)  # Sách ký hiệu BPSK

# Load image
im = imread('image2.png')

if im.shape[-1] == 4:
    im = im[:, :, :3]

im_bin = []
for i in range(3):  # Xử lý từng kênh màu
    channel = (im[:, :, i] * 255).astype(np.uint8) if im.dtype != np.uint8 else im[:, :, i]
    im_bin.append(np.unpackbits(channel.flatten()))

im_bin = np.concatenate(im_bin)  # Gộp các kênh lại

# Padding và ánh xạ ký hiệu
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=np.uint8)
im_bin_padded = np.concatenate((im_bin, padding))

# Chuyển đổi từ bit nhị phân sang ký hiệu BPSK
cons_sym_id = im_bin_padded  # Dữ liệu nhị phân chính là chỉ số ký hiệu trong BPSK
X = symbol_book[cons_sym_id]  # Ánh xạ vào sách ký hiệu

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
rec_syms = (X_hat.real >= 0).astype(np.uint8)  # 1 nếu >= 0, ngược lại là 0
rec_im_bin = rec_syms[:len(im_bin)]  # Lấy lại dữ liệu gốc
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
plt.savefig('result2_bpsk.png')
plt.show()
