import numpy as np
import matplotlib.pyplot as plt
import myDSP
from scipy.io.wavfile import write


fs = 48_000            # Hz
duration = 2.0         # secs
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 
# UsING 0.3 amplitude so the total sum stays under 1.0 to avoid digital clipping
amp = 0.3 
y1 = myDSP.mySine(amp,  800, 0.0, t)
y2 = myDSP.mySine(amp, 1200, 0.0, t)
y3 = myDSP.mySine(amp, 1600, 0.0, t)

y_sum = y1 + y2 + y3

# merged WAV
write('gen_signal_3tones.wav', fs, y_sum)
print('Saved merged sound as gen_signal_3tones.wav')

# visualizationnn
n_short = int(fs * 0.02)     # 960 samples
t_short = t[:n_short] * 1e3  # convert to ms

# Packing into lists for easy looping
signals = [
    (y1,    y1[:n_short],    '800 Hz Sine'),
    (y2,    y2[:n_short],    '1200 Hz Sine'),
    (y3,    y3[:n_short],    '1600 Hz Sine'),
    (y_sum, y_sum[:n_short], 'Sum 800+1200+1600 Hz')
]

# -Time Domain
fig1, axs1 = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for ax, (_, sig_short, title) in zip(axs1, signals):
    ax.plot(t_short, sig_short)
    ax.set_ylabel('Amplitude')
    ax.set_title(f"Time Domain (First 20 ms): {title}")
    ax.grid(True)

axs1[-1].set_xlabel('Time (ms)')
plt.tight_layout()

# Frequency Domain
def fft_mag(x, fs):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    return freqs, np.abs(X) / N

fig2, axs2 = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for ax, (sig_full, _, title) in zip(axs2, signals):
    f, M = fft_mag(sig_full, fs)
    ax.plot(f, M)
    ax.set_xlim(0, 3000) 
    ax.set_ylabel('Magnitude')
    ax.set_title(f"Frequency Domain: {title}")
    ax.grid(True)

axs2[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()

plt.show()