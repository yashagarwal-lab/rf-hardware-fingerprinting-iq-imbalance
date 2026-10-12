import numpy as np
import glob
import os
from scipy.signal import welch

fs = 400000.0

def get_sym_rate(x):
    # Find symbol rate by looking at spectrum of x^2 or magnitude
    mag = np.abs(x)
    F = np.fft.fft(mag - np.mean(mag))
    f = np.fft.fftfreq(len(mag), 1/fs)
    pos_idx = (f > 1000) & (f < 10000)
    peak_idx = np.argmax(np.abs(F[pos_idx]))
    return f[pos_idx][peak_idx]

def analyze_file(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    x = raw[0::2] + 1j*raw[1::2]
    
    # Power
    pwr = np.mean(np.abs(x)**2)
    
    # Calculate PSD
    f, Pxx = welch(x, fs, nperseg=16384, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    dc_mask_ignore = (f > -1000) & (f < 1000)
    Pxx_no_dc = Pxx.copy()
    Pxx_no_dc[dc_mask_ignore] = 0
    peak_idx = np.argmax(Pxx_no_dc)
    fc = f[peak_idx]
    
    # Symbol rate
    sym_rate = get_sym_rate(x)
    
    return fc, 10*np.log10(pwr), sym_rate

print("Reference Radios:")
ref_pat = "data/reference/*.sigmf-data" if os.path.exists("data/reference") else "reference/*.sigmf-data"
refs = sorted(glob.glob(ref_pat))
for ref in refs:
    fc, pwr, sym_rate = analyze_file(ref)
    name = os.path.basename(ref).split('.')[0]
    print(f"{name:12s}: fc = {fc:8.1f} Hz, pwr = {pwr:6.2f} dB, sym_rate = {sym_rate:6.1f} Hz")

print("\nUnknown:")
unk_f = "data/unknown/unknown_transmission.sigmf-data" if os.path.exists("data/unknown") else "unknown/unknown_transmission.sigmf-data"
unk_fc, unk_pwr, unk_sym = analyze_file(unk_f)
print(f"Unknown     : fc = {unk_fc:8.1f} Hz, pwr = {unk_pwr:6.2f} dB, sym_rate = {unk_sym:6.1f} Hz")
