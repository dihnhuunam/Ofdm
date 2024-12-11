import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve
import ofdm_demodulator
import ofdm_modulator
import qam_demodulator
import qam_modulator
import awgn
import mcm_channel_model
import symerr

# Rest of the code remains the same as in the original script
# ... [The rest of the implementation follows]


# Parameters for OFDM system and Monte Carlo channel model
NFFT = 64  # FFT size
G = 9  # Guard interval size
M_ary = 16  # QAM modulation order
t_a = 50e-9  # Symbol duration
rho = np.array([100, 0.6095, 0.4945, 0.3940, 0.2371, 0.19, 0.1159, 0.0699, 0.0462])  # Channel tap gains
N_P = len(rho)

# Parameters for Monte Carlo channel model
symbol_duration = NFFT * t_a
number_of_summations = 40
f_dmax = 50.0
NofOFDMSymbol = 1000
length_data = NofOFDMSymbol * NFFT

# Generate random source data
source_data = np.random.randint(0, M_ary, length_data)

# QAM modulation
qam_symbols = qam_modulator.qam_modulator(source_data, M_ary)

# Arrange QAM symbols into OFDM data pattern
data_pattern = qam_symbols.reshape((NofOFDMSymbol, NFFT))

# Monte Carlo simulations for multiple realizations
Number_Relz = 50
ser_relz = []

for number_of_realization in range(Number_Relz):
    # Generate random variables for channel realization
    u = np.random.rand(N_P, number_of_summations)
    ser = []
    snr_min = 0
    snr_max = 25
    step = 1
    snr_range = np.arange(snr_min, snr_max + 1, step)
    print(number_of_realization)
    for snr in snr_range:
        snt = snr - 10 * np.log10((NFFT + G) / NFFT)  # Effective SNR
        rs_frame = []
        h_frame = []
        initial_time = 0

        for i in range(NofOFDMSymbol):
            ofdm_signal = ofdm_modulator.ofdm_modulator(data_pattern[i, :], NFFT, G)
            h, t = mcm_channel_model.mcm_channel_model(u, initial_time, number_of_summations, symbol_duration, f_dmax, rho)
            h_frame.append(h)
            rs = convolve(ofdm_signal, h)[:len(ofdm_signal)]
            rs = awgn.awgn(rs, snt)
            rs_frame.append(rs)
            initial_time = t

        rs_frame = np.array(rs_frame)
        h_frame = np.array(h_frame)

        # Demodulate received symbols with MMSE equalization
        receiver_data = []
        data_symbol = []

        for i in range(NofOFDMSymbol):
            rs_i = rs_frame[i, :]
            demodulated_signal_i = ofdm_demodulator.ofdm_demodulator(rs_i, NFFT, NFFT, G)
            h = h_frame[i, :]
            H = fft(np.concatenate([h, np.zeros(NFFT - N_P)]))

            # MMSE Equalization
            noise_power = 10**(-snt / 10)  # N0 = Signal Power / SNR_linear
            MMSE = np.conj(H) / (np.abs(H)**2 + noise_power)
            d = demodulated_signal_i * MMSE

            demodulated_symbol_i = qam_demodulator.qam_demodulator(d, M_ary)
            data_symbol.extend(demodulated_symbol_i)

        data_symbol = np.array(data_symbol)

        num_errors, error_rate = symerr.symerr(source_data, data_symbol)
        ser.append(error_rate)

    ser_relz.append(ser)

# Average SER over all realizations
ser = np.mean(ser_relz, axis=0)
snr = snr_min + np.arange(len(ser)) * step

# Plot Symbol Error Rate (SER) vs SNR
plt.semilogy(snr, ser, 'bo')
plt.ylabel('SER')
plt.xlabel('SNR in dB')
plt.title('Symbol Error Rate (SER) vs SNR for OFDM with MMSE Equalization')
plt.grid(True)
plt.savefig('SER_vs_SNR_OFDM_MMSE.png')
plt.show()

# Parameters
N = 64  # Number of subcarriers
M = 16  # QAM order
snr_dB = 20  # SNR in dB
G = 9  # Guard interval size
rho = np.array([1, 0.8, 0.6, 0.4, 0.2])  # Channel taps
N_P = len(rho)

# Step 1: QAM Modulation
data = np.random.randint(0, M, N)  # Random data symbols
qam_symbols = np.sqrt(1 / 10) * np.exp(1j * (np.pi / M) * (2 * data + 1))  # QAM modulated symbols

plt.figure(figsize=(10, 5))
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), color='blue', label="QAM Symbols")
plt.title("Step 1: QAM Modulation - Constellation Diagram")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('step1_qam_modulation.png')
plt.show()

# Step 2: OFDM Modulation
ofdm_signal = np.concatenate([ifft(qam_symbols), ifft(qam_symbols)[-G:]])

plt.figure(figsize=(10, 5))
plt.plot(np.real(ofdm_signal), label="Real Part")
plt.plot(np.imag(ofdm_signal), label="Imaginary Part", linestyle='--')
plt.title("Step 2: OFDM Modulation - Time Domain Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig('step2_ofdm_modulation.png')
plt.show()

# Step 3: Transmit through Fading Channel
h = rho * (np.random.randn(N_P) + 1j * np.random.randn(N_P)) / np.sqrt(2)  # Channel taps
received_signal = np.convolve(ofdm_signal, h, mode='same')

plt.figure(figsize=(10, 5))
plt.plot(np.abs(h), 'o-', label="Channel Tap Magnitude")
plt.title("Step 3: Fading Channel - Channel Taps")
plt.xlabel("Tap Index")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.savefig('step3_fading_channel.png')
plt.show()

# Step 4: Add AWGN Noise
snr_linear = 10**(snr_dB / 10)
power_signal = np.mean(np.abs(received_signal)**2)
noise_variance = power_signal / snr_linear
noise = np.sqrt(noise_variance / 2) * (np.random.randn(*received_signal.shape) + 1j * np.random.randn(*received_signal.shape))
noisy_signal = received_signal + noise

plt.figure(figsize=(10, 5))
plt.plot(np.real(noisy_signal), label="Real Part")
plt.plot(np.imag(noisy_signal), label="Imaginary Part", linestyle='--')
plt.title("Step 4: Noisy Signal - After Fading and AWGN")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig('step4_awgn_noise.png')
plt.show()

# Step 5: MMSE Equalization
H = fft(np.concatenate([h, np.zeros(N - N_P)]))  # Channel Frequency Response
Y = fft(noisy_signal[:N])
MMSE = np.conj(H) / (np.abs(H)**2 + (1 / snr_linear))
equalized_signal = Y * MMSE

plt.figure(figsize=(10, 5))
plt.scatter(np.real(equalized_signal), np.imag(equalized_signal), color='red', label="Equalized Signal")
plt.title("Step 5: MMSE Equalization - Equalized Constellation")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('step5_mmse_equalization.png')
plt.show()