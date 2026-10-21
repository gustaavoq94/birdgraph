#!/usr/bin/env python3
"""
birdfeatures.py -- MFCC, chroma y PCA implementados a mano (solo numpy/scipy).

No uso librosa a proposito: asi el pipeline es transparente y se puede auditar
coeficiente por coeficiente, que es lo que importa si despues vas a comparar
especies estadisticamente.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


# --------------------------------------------------------------------------- mel
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 64,
                   fmin: float = 300.0, fmax: float = 12000.0) -> np.ndarray:
    """Banco de filtros triangulares en escala mel. Devuelve (n_mels, n_bins)."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    # normalizacion slaney: cada filtro con area equivalente
    enorm = 2.0 / (edges[2:n_mels + 2] - edges[:n_mels])
    return fb * enorm[:, None]


def mfcc(mag: np.ndarray, sr: int, n_fft: int, n_mfcc: int = 40, n_mels: int = 64,
         fmin: float = 300.0, fmax: float = 12000.0,
         deltas: bool = True) -> np.ndarray:
    """MFCC cuadro a cuadro desde la magnitud del STFT. Devuelve (n_frames, D).

    Codifica la envolvente timbrica; con deltas=True agrega la derivada temporal,
    que es la parte de 'dinamica espectral'.
    """
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = fb @ (mag.astype(np.float32) ** 2)                 # (n_mels, n_frames)
    logmel = np.log(mel + 1e-10)
    coef = dct(logmel, type=2, axis=0, norm="ortho")[:n_mfcc]   # (n_mfcc, n_frames)
    out = coef.T                                             # (n_frames, n_mfcc)
    if deltas:
        d = np.gradient(out, axis=0)
        out = np.concatenate([out, d], axis=1)
    return np.ascontiguousarray(out.astype(np.float32))


# --------------------------------------------------------------------------- chroma
def chroma(mag: np.ndarray, sr: int, n_fft: int,
           fmin: float = 300.0, fmax: float = 12000.0) -> np.ndarray:
    """Energia plegada en 12 clases de altura. Devuelve (12, n_frames) normalizado."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[band]
    pc = np.round(12 * np.log2(f / 440.0) + 69).astype(int) % 12
    e = mag[band] ** 2
    out = np.zeros((12, mag.shape[1]), dtype=np.float32)
    for k in range(12):
        m = pc == k
        if m.any():
            out[k] = e[m].sum(axis=0)
    return out / (out.sum(axis=0, keepdims=True) + 1e-12)


def chroma_concentration(ch: np.ndarray) -> np.ndarray:
    """Que tan concentrada esta la energia en pocas clases (0 = plano, 1 = una sola).
    Es 1 - entropia normalizada; sube en cantos tonales y baja en ruido."""
    p = np.clip(ch, 1e-12, None)
    h = -(p * np.log(p)).sum(axis=0) / np.log(12.0)
    return np.clip(1.0 - h, 0, 1)


# --------------------------------------------------------------------------- PCA
class PCA3:
    """PCA por SVD. Se ajusta solo con los cuadros activos (con senal) para que el
    silencio no domine los ejes, y despues proyecta todo."""

    def __init__(self, X: np.ndarray, active: np.ndarray | None = None,
                 whiten: bool = True, standardize: bool = False):
        Xf = X[active] if active is not None and active.any() else X
        self.mu = Xf.mean(axis=0)
        # por defecto solo centro: los MFCC ya estan en unidades comparables y
        # estandarizar le da el mismo peso a los coeficientes altos, que son ruido
        self.sd = (Xf.std(axis=0) + 1e-8) if standardize else np.ones(X.shape[1], np.float32)
        Z = (Xf - self.mu) / self.sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        self.comp = Vt[:3]                                   # (3, D)
        n = max(len(Z) - 1, 1)
        self.explained = (S[:3] ** 2 / n) / ((S ** 2 / n).sum() + 1e-12)
        self.scale = (S[:3] / np.sqrt(n)) if whiten else np.ones(3)
        self.scale = np.maximum(self.scale, 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Y = Z @ self.comp.T
        return (Y / self.scale).astype(np.float32)


def smooth_traj(Y: np.ndarray, k: int = 5) -> np.ndarray:
    """Media movil sobre la trayectoria: la vuelve continua y legible."""
    if k <= 1:
        return Y
    ker = np.ones(k) / k
    return np.stack([np.convolve(Y[:, i], ker, mode="same") for i in range(Y.shape[1])],
                    axis=1).astype(np.float32)
