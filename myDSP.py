import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundcard as sc
from scipy.io import wavfile
from scipy.signal.windows import hann



def mySine(A, f, theta0, t):
    return A * np.sin(2 * np.pi * f * t) + theta0

def myCosine(A, f, theta0, t):
    return A * np.cos(2 * np.pi * f * t) + theta0

def myRamp(A, t1, t):
    return A * (t >= t1)

def myBox(A, t1, t2, t):
    return A * ((t >= t1) & (t < t2))

def myWhiteNoise(A, t):
    n = np.random.rand(np.size(t))
    return 2 * A * (n - n.mean())

def myDFT(x):
    N = np.size(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def plotInTime(x, fs):
    t = np.arange(len(x)) / fs
    if x.ndim == 1:  # mono
        plt.plot(t, x, label='Mono')
    elif x.ndim == 2:  # stereo
        plt.plot(t, x[:, 0], label='Left Channel', color='blue', alpha=0.7)
        plt.plot(t, x[:, 1], label='Right Channel', color='red', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)


def plotInTimeHann(x, fs):
    t = np.arange(len(x)) / fs
    

    window = hann(len(x))

    if x.ndim == 1:  # Mono
        x_win = x * window
        plt.plot(t, x_win, label='Mono')
    elif x.ndim == 2:  # Stereo
        x_win_left = x[:, 0] * window
        x_win_right = x[:, 1] * window
        plt.plot(t, x_win_left, label='Left Channel', color='blue', alpha=0.7)
        plt.plot(t, x_win_right, label='Right Channel', color='red', alpha=0.7)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
def plotInFrequency(x, fs):
    N = int(np.size(x, 0) / 2)
    
    if np.size(x, 0) == 1:
        X = np.fft.fft(x,n=N, axis=1)
    else:
        X = np.fft.fft(x, n=N,axis=0)
    X = np.abs(X)
    X = X[:N//2]
    f = np.arange(0, fs / 2, fs / N)

    plt.plot(f, X, marker='*')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
def plotInFrequencyHann(x, fs):
    N = int(np.size(x, 0) / 2)
    window = hann(len(x))
    x_win = x * window
    
    if np.size(x, 0) == 1:
        X = np.fft.fft(x_win,n=N, axis=1)
    else:
        X = np.fft.fft(x_win, n=N,axis=0)
    X = np.abs(X)
    X = X[:N//2]
    step = fs / N
    num_bins = N // 2
    f = np.arange(0, num_bins * step, step)
    
    plt.plot(f, X, marker='*')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

def play(x, fs):
    sd.play(x, fs)

def record(t, fs):
    mike = sc.default_microphone()
    samples = fs * t
    x = mike.record(samples, fs)
    return x

def readWav(fileName):
    return wavfile.read(fileName)
