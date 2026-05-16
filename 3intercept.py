import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.fft import fft
from scipy.io import wavfile

# 1. User Parameters based on YOUR files
VOLUME_LEVELS = [10, 30, 50, 70, 90]
FS_EXPECTED = 48_000
F1, F2, F3 = 800, 1200, 1600
START_SEC, END_SEC = 1.0, 1.25 # 0.25s slice logic [cite: 94, 977]

# Frequency Selection for Analysis:
# Even Order: f2 - f1 = 400 Hz (Clean)
# Odd Order (5th): 3*f2 - 2*f1 = 2000 Hz (Clean)
F_EVEN = 400
F_ODD  = 2000

# 2. Helper Functions (Exact Colleague Logic) [cite: 993-1000]
def measure_db(sig, fs, target_freq):
    win = hann(len(sig))
    nfft = len(sig) // 2
    X = fft(sig * win, n=nfft)[:nfft // 2]
    mags = np.abs(X)
    freqs = np.linspace(0, fs / 2, len(mags), endpoint=False)
    idx = np.argmin(np.abs(freqs - target_freq))
    return 20 * np.log10(mags[idx] + 1e-12)

# 3. Gather Measured Levels from your "laptoprec" files
vols, fund_db, even_db, odd_db = [], [], [], []

for pct in VOLUME_LEVELS:
    filename = f"laptoprec{pct}.wav"
    if not os.path.isfile(filename):
        print(f"Skipping {filename}: File not found")
        continue
    
    fs, data = wavfile.read(filename)
    sig = data[:, 0] if data.ndim == 2 else data
    seg = sig[int(START_SEC * fs): int(END_SEC * fs)]
    
    vols.append(pct)
    fund_db.append(measure_db(seg, fs, F2))
    even_db.append(measure_db(seg, fs, F_EVEN))
    odd_db.append(measure_db(seg, fs, F_ODD))

# 4. Calculate Intercepts and Plot
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
analysis_targets = [(even_db, "Even (400Hz)"), (odd_db, "Odd (2000Hz)")]

for ax, (dist_db, label) in zip(axs, analysis_targets):
    # Fit lines using ONLY the first three points [cite: 1045, 1052-1056]
    # We use Fundamental dB as the X-axis instead of Percent to stabilize slopes.
    x_fit = np.array(fund_db[:3])  # The "Input" power
    y_f   = np.array(fund_db[:3])  # Fundamental vs itself (Slope will be exactly 1.0)
    y_d   = np.array(dist_db[:3])  # Distortion vs Fundamental
    
    # Linear fits using only the first three points [cite: 922, 1045]
    m_f, b_f = np.polyfit(x_fit, y_f, 1)
    m_d, b_d = np.polyfit(x_fit, y_d, 1)
    
    # Compute intercept in dB [cite: 1058-1059]
    db_int = (b_d - b_f) / (m_f - m_d)
    y_int = m_f * db_int + b_f
    
    # Plotting adjustments
    x_line = np.linspace(min(fund_db), db_int + 5, 500)
    ax.plot(fund_db, fund_db, 'ob', label=f"Fundamental @ {F2}Hz")
    ax.plot(fund_db, dist_db, 'sr', label=f"{label} Distortion")
    ax.plot(x_line, m_f * x_line + b_f, '--b', alpha=0.6)
    ax.plot(x_line, m_d * x_line + b_d, '--r', alpha=0.6)
    ax.plot(db_int, y_int, 'kx', markersize=12, markeredgewidth=2, label=f"IIP Intercept @ {db_int:.1f} dB")
    
    ax.set_title(f"{label} Order Intercept Analysis")
    ax.set_ylabel("Level (dB)")
    ax.grid(True, linestyle=':')
    ax.legend()

axs[-1].set_xlabel("Playback Level (%)")
plt.tight_layout()
plt.show()

print(f"IIP Even Intercept: { ( (even_db[0]-fund_db[0]) / (mf-md) ) if 'mf' in locals() else 'Check Plot' }")