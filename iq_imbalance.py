import numpy as np
import glob
import os

fs = 400000.0

def complex_imbalance(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    x = raw[0::2] + 1j*raw[1::2]
    
    # 1. Precise fc using x^2
    x2 = x**2
    F2 = np.fft.fft(x2)
    freqs2 = np.fft.fftfreq(len(x2), 1/fs)
    pos_idx = np.abs(freqs2) > 1000
    peak_idx = np.argmax(np.abs(F2[pos_idx]))
    fc = freqs2[pos_idx][peak_idx] / 2.0
    
    # 2. Extract K
    F = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), 1/fs)
    
    # Find bins around fc
    bw = 4000
    sig_mask = (freqs > fc - bw) & (freqs < fc + bw)
    
    # Find bins around -fc
    img_mask = (freqs > -fc - bw) & (freqs < -fc + bw)
    
    # Extract the slices
    # To properly match them, we need to sort by frequency
    sig_freqs = freqs[sig_mask]
    sig_bins = F[sig_mask]
    
    img_freqs = freqs[img_mask]
    img_bins = F[img_mask]
    
    # Sort them so they align
    sig_sort = np.argsort(sig_freqs)
    sig_bins = sig_bins[sig_sort]
    
    img_sort = np.argsort(img_freqs)
    # The image of S(f - fc) at -fc is S(f + fc). Wait, S(-f + fc) ?
    # Since s(t) is real, S(-f) = S*(f).
    # X(f) = S(f - fc) + K S(f + fc).
    # So X(f + fc) = K S(f + 2fc). Wait, around -fc:
    # X(f_img) for f_img near -fc corresponds to K S(f_img + fc).
    # X(f_sig) for f_sig near +fc corresponds to S(f_sig - fc).
    # Let delta = f_sig - fc. Then f_img = -fc + delta.
    # So X(-fc + delta) = K S(delta).
    # X(fc + delta) = S(delta).
    # Therefore, X(-fc + delta) = K * X(fc + delta).
    # So we just match f_img and f_sig such that f_img - (-fc) = f_sig - fc.
    # i.e., f_img = f_sig - 2fc.
    
    # Let's just find the corresponding bins
    K_num = 0.0 + 0.0j
    K_den = 0.0
    
    for f_s, bin_s in zip(sig_freqs, F[sig_mask]):
        f_i = f_s - 2*fc
        # find the bin in F closest to f_i
        idx = np.argmin(np.abs(freqs - f_i))
        if np.abs(freqs[idx] - f_i) < 1.0: # Exact match
            bin_i = F[idx]
            K_num += bin_i * np.conj(bin_s)
            K_den += np.abs(bin_s)**2
            
    K = K_num / K_den if K_den > 0 else 0
    
    irr = -10 * np.log10(np.abs(K)**2) if np.abs(K) > 0 else 999
    angle = np.angle(K, deg=True)
    
    return fc, irr, angle

print("Reference Radios:")
ref_pat = "data/reference/*.sigmf-data" if os.path.exists("data/reference") else "reference/*.sigmf-data"
refs = sorted(glob.glob(ref_pat))
for ref in refs:
    fc, irr, angle = complex_imbalance(ref)
    name = os.path.basename(ref).split('.')[0]
    print(f"{name:12s}: fc = {fc:8.1f} Hz, IRR = {irr:6.2f} dB, Angle = {angle:6.1f} deg")

print("\nUnknown:")
unk_fc, unk_irr, unk_angle = complex_imbalance("data/unknown/unknown_transmission.sigmf-data" if os.path.exists("data/unknown") else "unknown/unknown_transmission.sigmf-data")
print(f"Unknown     : fc = {unk_fc:8.1f} Hz, IRR = {unk_irr:6.2f} dB, Angle = {unk_angle:6.1f} deg")
