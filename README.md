# RF Hardware Fingerprinting via IQ Imbalance and Image Rejection Ratio

> **tl;dr:** Every physical radio transmitter has manufacturing imperfections in its IQ mixer that produce a unique spectral "ghost image." This Image Rejection Ratio (IRR) is a hardware fingerprint that cannot be spoofed in software. This repo demonstrates how to extract IRR from SigMF captures and identify an unknown transmitter — even when it perfectly mimics another radio's carrier offset, power, and baud rate.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data via LFS](https://img.shields.io/badge/data-Git%20LFS-orange)
![Topics](https://img.shields.io/badge/topics-RF%20Fingerprinting%20%7C%20IQ%20Imbalance%20%7C%20IRR%20%7C%20Emitter%20ID-purple)

---

## 1. The Engineering Problem

Eight reference radios (A–H) have been catalogued. A rogue operator intercepts the reference data and clones Radio X's carrier frequency offset, output power, and symbol rate exactly. An unknown transmission is recorded. A standard spectrum comparison identifies it as Radio X — but Radio X's operator denies transmitting.

The goal is to identify the **true** transmitter using a hardware property the operator cannot change.

---

## 2. The Physics of IQ Imbalance

A radio transmitter generates RF by mixing a baseband IQ signal with a Local Oscillator (LO) at the carrier frequency:

```
RF(t) = I(t)·cos(2π·f_LO·t) - Q(t)·sin(2π·f_LO·t)
```

In a real analog IQ mixer, the two branches are never perfectly matched. Two types of imperfection exist:

**Amplitude Imbalance (ε):** The Q branch has gain ε ≠ 1 relative to I:
```
RF(t) = I(t)·cos(2π·f_LO·t) - ε·Q(t)·sin(2π·f_LO·t)
```

**Phase Imbalance (φ):** The LO quadrature is not exactly 90°:
```
RF(t) = I(t)·cos(2π·f_LO·t) - ε·Q(t)·sin(2π·f_LO·t + φ)
```

Using Euler's formula, imperfect mixing produces both the desired signal at `+f_LO` **and an image signal at `-f_LO`**.

---

## 3. The Mathematical Foundation

### 3.1 Image Rejection Ratio (IRR)

The IRR is defined as the power ratio between the image and the main signal, in dB:

```
IRR_dB = 10·log₁₀(P_image / P_main)
       = 20·log₁₀(|ε·e^(jφ) - 1| / |ε·e^(jφ) + 1|)
```

For a perfect mixer: ε = 1, φ = 0 → IRR = -∞ dB (no image).
For a real radio: IRR is typically between -20 dB and -60 dB depending on hardware quality.

Each physical radio has a fixed, repeatable IRR determined by its analog component tolerances. **This is as unique as a fingerprint.**

### 3.2 IRR Extraction via FFT

```python
X  = np.fft.fft(iq_burst)           # FFT of a clean signal burst
f  = np.fft.fftfreq(len(iq_burst), d=1/sample_rate)

# Find the main carrier peak at +f_c
main_bin   = np.argmax(np.abs(X[f > 0]))
P_main_dB  = 20*np.log10(np.abs(X[main_bin]))

# Measure the image at -f_c (symmetric bin)
image_bin  = len(X) - main_bin
P_image_dB = 20*np.log10(np.abs(X[image_bin]))

IRR_dB = P_image_dB - P_main_dB
```

### 3.3 Identification Strategy

1. Catalogue `IRR_dB` for each reference radio (averaged over 3 captures)
2. Extract `IRR_dB` from the unknown transmission
3. The unknown's carrier/power/baud match Radio X → obvious match
4. The unknown's `IRR_dB` matches Radio Y → Radio Y is the true transmitter

The impersonator cannot fake IRR by adjusting transmit parameters — it is set by physics.

---

## 4. Dataset — SigMF Schema

| File | Size | Format | Description |
|------|------|--------|-------------|
| `data/reference/radio_A_1.sigmf-data` ... `radio_H_3.sigmf-data` | 24 × 313 KB | `cf32_le` | 3 captures each for 8 radios |
| `data/unknown/unknown_transmission.sigmf-data` | 938 KB | `cf32_le` | Unknown emitter to identify |
| All `*.sigmf-meta` | ~300 B each | JSON | Frequency, sample rate metadata |

```python
import numpy as np
raw = np.fromfile("data/reference/radio_A_1.sigmf-data", dtype=np.float32)
iq  = raw[0::2] + 1j * raw[1::2]
```

---

## 5. Repository Structure

```
rf-hardware-fingerprinting-iq-imbalance/
├── data/
│   ├── reference/
│   │   ├── radio_A_1.sigmf-data  ← LFS (×24 files)
│   │   └── ...
│   └── unknown/
│       └── unknown_transmission.sigmf-data  ← LFS
├── solve.py              # Impersonation analyser (primary solver)
├── iq_imbalance.py       # IQ amplitude and phase imbalance quantifier
├── peak_irr.py           # Fast FFT-based IRR peak finder
└── precise_irr.py        # High-resolution IRR with windowed FFT
```

---

## 6. Setup and Execution

```bash
git clone https://github.com/<username>/rf-hardware-fingerprinting-iq-imbalance.git
cd rf-hardware-fingerprinting-iq-imbalance
git lfs pull

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python solve.py
```

---

## 7. Further Reading

- Danev et al. — "Physical-layer Identification of RFID Devices", *USENIX Security*, 2009
- Brik et al. — "Wireless Device Identification with Radiometric Signatures", *MobiCom*, 2008
- Pozar — *Microwave Engineering*, 4th Ed., Ch. 10 (Mixer design and image rejection)
- ITU-R SM.1600 — *Technical identification of digital signals*
