import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import convolve
from skimage.io import imread

# Thiết lập các thông số cơ bản
mod_method = 'QPSK'
n_fft = 256          # Số điểm FFT
n_cpe = 32          # Độ dài tiền tố cyclic
snr = 30            # Tỷ lệ tín hiệu trên nhiễu (dB)
n_taps = 4          # Số tap của kênh fading
ch_est_method = 'MMSE'

# Cấu hình điều chế
mod_methods = ['BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM']
mod_order = mod_methods.index(mod_method) + 1

# Đọc và xử lý ảnh đầu vào
im = imread('./images/logo.png')
if im.shape[-1] == 4:
    im = im[:, :, :3]

# Chuyển đổi ảnh sang dãy bit nhị phân
im_bin = []
for i in range(3):
    channel = (im[:, :, i] * 255).astype(np.uint8) if im.dtype != np.uint8 else im[:, :, i]
    im_bin.append(np.unpackbits(channel.flatten()))
im_bin = np.concatenate(im_bin)

# Tạo bảng ký hiệu QPSK
mod_ind = 2 ** mod_order
n = np.arange(0, 2 * np.pi, 2 * np.pi / mod_ind)
symbol_book = np.exp(1j * n)

# Thêm padding và ánh xạ dữ liệu sang ký hiệu điều chế
sym_rem = (mod_order - len(im_bin) % mod_order) % mod_order
padding = np.zeros(sym_rem, dtype=np.uint8)
im_bin_padded = np.concatenate((im_bin, padding))
cons_data = im_bin_padded.reshape(-1, mod_order)
cons_sym_id = np.dot(cons_data, 1 << np.arange(cons_data.shape[1])[::-1])
cons_sym_id = cons_sym_id % len(symbol_book)

# Điều chế dữ liệu
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
rec_syms = np.array([np.argmin(np.abs(symbol_book - sym)) for sym in X_hat])
rec_syms_cons = np.unpackbits(rec_syms.astype(np.uint8).reshape(-1, 1), axis=1)[:, -mod_order:]
rec_im_bin = rec_syms_cons.flatten()[:len(im_bin)]
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
plt.savefig('./results/result_qpsk.png')
plt.show()