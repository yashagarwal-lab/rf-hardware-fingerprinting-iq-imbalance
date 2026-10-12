import numpy as np
import glob
import os
from scipy.signal import welch

fs = 400000.0

def exact_irr(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    x = raw[0::2] + 1j*raw[1::2]
    
    f, Pxx = welch(x, fs, nperseg=32768, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    dc_mask_ignore = (f > -1000) & (f < 1000)
    Pxx_no_dc = Pxx.copy()
    Pxx_no_dc[dc_mask_ignore] = 0
    peak_idx = np.argmax(Pxx_no_dc)
    fc = f[peak_idx]
    
    bw = 6000.0
    sig_mask = (f > fc - bw) & (f < fc + bw)
    img_mask = (f > -fc - bw) & (f < -fc + bw)
    
    sig_pwr = np.sum(Pxx[sig_mask])
    img_pwr = np.sum(Pxx[img_mask])
    
    irr = 10 * np.log10(sig_pwr / img_pwr) if img_pwr > 0 else 999
    
    return fc, irr

print("Reference Radios:")
ref_pat = "data/reference/*.sigmf-data" if os.path.exists("data/reference") else "reference/*.sigmf-data"
refs = sorted(glob.glob(ref_pat))
for ref in refs:
    fc, irr = exact_irr(ref)
    name = os.path.basename(ref).split('.')[0]
    print(f"{name:12s}: fc = {fc:8.1f} Hz, IRR = {irr:6.3f} dB")

print("\nUnknown:")
unk_fc, unk_irr = exact_irr("data/unknown/unknown_transmission.sigmf-data" if os.path.exists("data/unknown") else "unknown/unknown_transmission.sigmf-data")
print(f"Unknown     : fc = {unk_fc:8.1f} Hz, IRR = {unk_irr:6.3f} dB")
