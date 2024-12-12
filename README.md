# Mô phỏng Truyền dẫn OFDM với Điều chế PSK/QAM

Dự án này mô phỏng quá trình truyền dẫn ảnh sử dụng kỹ thuật OFDM (Orthogonal Frequency Division Multiplexing) kết hợp với các phương pháp điều chế PSK/QAM.

## Tính năng

- Hỗ trợ nhiều phương pháp điều chế: BPSK, QPSK
- Mô phỏng kênh fading đa đường
- Ước lượng và cân bằng kênh sử dụng phương pháp MMSE
- Thêm nhiễu Gaussian trắng (AWGN)
- Hiển thị kết quả trực quan: sơ đồ chòm sao, ảnh gốc và ảnh khôi phục
- Tính toán tỉ lệ lỗi bit (BER)

## Yêu cầu

```txt
numpy
matplotlib
scipy
scikit-image
```

## Cách sử dụng

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Đặt ảnh cần truyền trong thư mục images/
3. Chạy mô phỏng:

```bash
python ofdm_BPSK.py  # cho điều chế BPSK
python ofdm_QBSK.py  # cho điều chế QPSK
```

## Cấu hình tham số

### Các tham số chính có thể điều chỉnh

- mod_method: Phương pháp điều chế (BPSK/QPSK)
- n_fft: Số điểm FFT
- n_cpe: Độ dài tiền tố cyclic
- snr: Tỉ lệ tín hiệu trên nhiễu (dB)
- n_taps: Số tap của kênh fading
- ch_est_method: Phương pháp ước lượng kênh

## Kết quả

### Kết quả mô phỏng được lưu trong thư mục results/, bao gồm

- Sơ đồ chòm sao tín hiệu phát
- Sơ đồ chòm sao tín hiệu thu
- Ảnh gốc và ảnh khôi phục
- Tỉ lệ lỗi bit (BER)
