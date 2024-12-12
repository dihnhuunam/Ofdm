import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage.io import imread

# Thiết lập các thông số cơ bản
mod_method = 'BPSK'
n_fft = 128         # Số điểm FFT
n_cpe = 16          # Độ dài tiền tố cyclic
snr = 30            # Tỷ lệ tín hiệu trên nhiễu (dB)
n_taps = 8          # Số tap của kênh fading
ch_est_method = 'MMSE'

# Cấu hình điều chế BPSK
mod_order = 1  # BPSK dùng 1 bit/ký hiệu
symbol_book = np.array([-1, 1], dtype=np.complex64)

# Đọc và xử lý ảnh đầu vào
im = imread('./images/image.png')
if im.shape[-1] == 4:
    im = im[:, :, :3]

# Chuyển đổi ảnh sang dãy bit nhị phân
im_bin = []
for i in range(3):
    channel = (im[:, :, i] * 255).astype(np.uint8) if im.dtype != np.uint8 else im[:, :, i]
    im_bin.append(np.unpackbits(channel.flatten()))
im_bin = np.concatenate(im_bin)

# Thêm padding và ánh xạ sang ký hiệu BPSK
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=np.uint8)
im_bin_padded = np.concatenate((im_bin, padding))
cons_sym_id = im_bin_padded
X = symbol_book[cons_sym_id]

# Xử lý OFDM: Tạo khối và thêm tiền tố cyclic
fft_rem = (n_fft - len(X) % n_fft) % n_fft
X_padded = np.concatenate((X, np.zeros(fft_rem, dtype=np.complex64)))
X_blocks = X_padded.reshape(-1, n_fft)
x = np.fft.ifft(X_blocks, axis=1)
x_cpe = np.hstack((x[:, -n_cpe:], x))
x_s = x_cpe.flatten()

# Thêm nhiễu vào tín hiệu
data_pwr = np.mean(np.abs(x_s)**2)
noise_pwr = data_pwr / (10**(snr / 10))
noise = np.sqrt(noise_pwr / 2) * (np.random.randn(*x_s.shape) + 1j * np.random.randn(*x_s.shape))
x_s_noise = x_s + noise

# Mô phỏng kênh fading
g = np.exp(-np.arange(n_taps))
g /= np.linalg.norm(g)
x_s_noise_fading = convolve(x_s_noise, g, mode='same')

# Khôi phục tín hiệu OFDM
x_p = x_s_noise_fading.reshape(-1, n_fft + n_cpe)
x_p_cpr = x_p[:, n_cpe:]
X_hat_blocks = np.fft.fft(x_p_cpr, axis=1)

# Ước lượng và cân bằng kênh
if n_taps > 1 and ch_est_method == 'MMSE':
    G = X_hat_blocks[:, 0] / X_blocks[:, 0]
    G_mmse = np.conj(G) / (np.abs(G)**2 + (noise_pwr / (data_pwr * 0.5)))
    X_hat_blocks *= G_mmse[:, None]

X_hat = X_hat_blocks.flatten()[:len(X)]

# Giải điều chế và khôi phục dữ liệu
rec_syms = (X_hat.real >= 0).astype(np.uint8)
rec_im_bin = rec_syms[:len(im_bin)]
ber = np.sum(rec_im_bin != im_bin) / len(im_bin)

# Khôi phục ảnh từ dữ liệu nhị phân
rec_im_bin_split = np.split(rec_im_bin, 3)
rec_channels = []
for i in range(3):
    rec_channels.append(np.packbits(rec_im_bin_split[i]).reshape(im[:, :, i].shape))
rec_im = np.stack(rec_channels, axis=-1)

# Hiển thị kết quả
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(X.real, X.imag, 'x', linewidth=2, markersize=10)
plt.xlim([-2, 2]); plt.ylim([-2, 2])
plt.xlabel('In phase'); plt.ylabel('Quadrature')
plt.title(f'Transmit Constellation\n{mod_method} Modulation')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(X_hat.real[::500], X_hat.imag[::500], 'x', markersize=3)
plt.xlim([-2, 2]); plt.ylim([-2, 2])
plt.xlabel('In phase'); plt.ylabel('Quadrature')
plt.title(f'Received Constellation\nMeasured SNR: {snr:.2f} dB')
plt.grid()

plt.subplot(2, 2, 3)
plt.imshow(im, cmap='gray')
plt.title('Transmit Image')

plt.subplot(2, 2, 4)
plt.imshow(rec_im, cmap='gray')
plt.title(f'Recovered Image\nBER: {ber:.2g}')

plt.tight_layout()
plt.savefig('./results/result_bpsk.png')
plt.show()