import numpy as np
import scipy.signal

fs = 400000.0

def peak_irr(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    x = raw[0::2] + 1j*raw[1::2]
    
    f, Pxx = scipy.signal.welch(x, fs, nperseg=65536, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    dc_mask_ignore = (f > -1000) & (f < 1000)
    Pxx_no_dc = Pxx.copy()
    Pxx_no_dc[dc_mask_ignore] = 0
    peak_idx = np.argmax(Pxx_no_dc)
    fc = f[peak_idx]
    
    # Find peak near fc
    sig_mask = (f > fc - 1000) & (f < fc + 1000)
    sig_peak = np.max(Pxx[sig_mask])
    
    # Find peak near -fc
    img_mask = (f > -fc - 1000) & (f < -fc + 1000)
    img_peak = np.max(Pxx[img_mask])
    
    irr = 10 * np.log10(sig_peak / img_peak)
    return irr

print("Radio D:")
print(peak_irr(("data/reference/radio_D_" if os.path.exists("data/reference") else "reference/radio_D_"1.sigmf-data"))
print(peak_irr(("data/reference/radio_D_" if os.path.exists("data/reference") else "reference/radio_D_"2.sigmf-data"))
print(peak_irr(("data/reference/radio_D_" if os.path.exists("data/reference") else "reference/radio_D_"3.sigmf-data"))
print("Unknown:")
print(peak_irr(("data/unknown/unknown_transmission.sigmf-data" if os.path.exists("data/unknown") else "unknown/unknown_transmission.sigmf-data")))
