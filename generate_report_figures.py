import os
import numpy as np
import matplotlib.pyplot as plt
import myDSP
from scipy.io.wavfile import read

# 1. Configuration
VOLUMES = [10, 30, 50, 70, 90]
FS = 48000
START_SEC, END_SEC = 1.0, 1.25 # The specific slice for the 2x2 grid
TOTAL_DURATION = 2.0           # For the 5-volume envelope plot

# --- FIGURE A: The 5-Volume Envelope Plot ---
fig_env, axs_env = plt.subplots(len(VOLUMES), 1, figsize=(10, 12), sharex=True)
print("Generating 5-Volume Envelope plot...")

for i, pct in enumerate(VOLUMES):
    filename = f"laptoprec{pct}.wav"
    fs, data = read(filename)
    sig = data[:, 0] if data.ndim == 2 else data
    
    t = np.arange(len(sig)) / fs
    axs_env[i].plot(t, sig, linewidth=0.5)
    axs_env[i].set_title(f"{pct}% System Volume Envelope")
    axs_env[i].set_ylabel("Amplitude")
    axs_env[i].grid(True, alpha=0.3)

axs_env[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("Figure_Volume_Envelopes.png", dpi=300) # Save high-quality for report
plt.show()

# --- FIGURE B: The 2x2 Analysis Grid (One for each file) ---
for pct in VOLUMES:
    filename = f"laptoprec{pct}.wav"
    print(f"Generating 2x2 analysis for {filename}...")
    
    fs, data = read(filename)
    sig = data[:, 0] if data.ndim == 2 else data
    
    # Extract the steady-state slice [cite: 830-831]
    sig_seg = sig[int(START_SEC * fs):int(END_SEC * fs)]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    base_title = f"{pct}% Volume Analysis"
    
    # 1) Time Domain (No Window) [cite: 833-836]
    plt.sca(axs[0, 0])
    myDSP.plotInTime(sig_seg, fs)
    plt.title(f"Time Domain (No Window): {base_title}")
    
    # 2) Time Domain (Hann Window) [cite: 837-840]
    plt.sca(axs[0, 1])
    myDSP.plotInTimeHann(sig_seg, fs)
    plt.title(f"Time Domain (Hann Window): {base_title}")
    
    # 3) Frequency Domain (No Window) [cite: 841-844]
    plt.sca(axs[1, 0])
    myDSP.plotInFrequency(sig_seg, fs)
    plt.xlim(0, 3000) # Zoomed to see our tones clearly
    plt.title("FFT Spectrum (No Window)")
    
    # 4) Frequency Domain (Hann Window) [cite: 845-848]
    plt.sca(axs[1, 1])
    myDSP.plotInFrequencyHann(sig_seg, fs)
    plt.xlim(0, 3000)
    plt.title("FFT Spectrum (Hann Window)")
    
    plt.tight_layout()
    plt.savefig(f"Analysis_{pct}pct.png", dpi=300)
    plt.show()