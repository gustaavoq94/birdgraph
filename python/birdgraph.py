#!/usr/bin/env python3
"""
birdgraph.py -- Visualizacion tipo "sistema visual + modelo matematico" de cantos de aves.

Idea: en vez de un espectrograma plano, se extraen los PICOS espectrales (parciales)
cuadro a cuadro, se ubican en un espacio (tiempo, log-frecuencia) y se conectan como
un GRAFO:

  - cadenas de parciales (partial tracking)  -> lineas magenta/rojas (estructura tonal local)
  - stack armonico del pico dominante        -> abanico verde fosforescente
  - arbol de expansion minima (MST)          -> hilos blancos finos que unen todo

Ademas se calculan descriptores por cuadro (centroide, spread, flux, tonalidad/flatness)
que se muestran como lectura numerica, y quedan exportables a CSV para comparacion
entre especies / analisis estadistico.

Uso:
    python birdgraph.py entrada.mp4 --mode still  --out frame.png --at 2.5
    python birdgraph.py entrada.wav --mode video  --out out.mp4 --fps 30
    python birdgraph.py entrada.wav --mode csv    --out features.csv

Requisitos: numpy, scipy, matplotlib  (+ ffmpeg en el PATH para video/mp3/mp4)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, get_window
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.colors import to_rgba

# --------------------------------------------------------------------------------------
# CONFIG RAPIDA  <<<<<< EDITA ACA SI NO QUERES USAR LA LINEA DE COMANDOS
# --------------------------------------------------------------------------------------
# Si INPUT no esta vacio y ejecutas el script sin argumentos (Run en VS Code),
# se usan estos valores. Si pasas argumentos por consola, estos se ignoran.
INPUT = r""                      # ej: r"C:\ruta\a\tu\pajaro.mp4"
DEFAULTS = [
    "--mode", "still",           # still | video | csv
    "--out",  "salida.png",      # .png para still, .mp4 para video, .csv para csv
    # "--at", "4.05",            # segundo a renderizar (still); sin esto usa el pico mas fuerte
    # "--start", "0", "--end", "20",
    # "--fps", "30",
    "--window", "0.9",
]

# --------------------------------------------------------------------------------------
# PALETA  (fosforescente sobre negro)
# --------------------------------------------------------------------------------------
BG        = "#0b0b0d"
C_WEB     = "#cfd6dd"   # hilos MST, blanco frio
C_PARTIAL = "#ff2f7a"   # cadenas de parciales, magenta
C_PARTIAL2= "#ff7a4d"   # parciales de baja energia, naranja
C_HARM    = "#25ff9a"   # abanico armonico, verde fosforo
C_HOT     = "#ffffff"   # nodo dominante
C_ISO     = "#f2ff45"   # nodo aislado, amarillo
C_TXT     = "#9aa3ab"
C_HUD     = "#e8eef4"
C_ACC     = "#ff3355"

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------------------
# 1. AUDIO IN  (equivalente al Audio File In CHOP)
# --------------------------------------------------------------------------------------
def load_audio(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Decodifica CUALQUIER archivo (wav/mp3/flac/mp4/mkv/mov) a mono float32 via ffmpeg."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".wav"):
        try:
            from scipy.io import wavfile
            file_sr, data = wavfile.read(path)
            x = data.astype(np.float32)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if np.issubdtype(data.dtype, np.integer):
                x /= float(np.iinfo(data.dtype).max)
            if file_sr != sr:
                n = int(round(len(x) * sr / file_sr))
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            return x.astype(np.float32), sr
        except Exception:
            pass  # cae a ffmpeg

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Necesito ffmpeg en el PATH para leer este formato.")

    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


# --------------------------------------------------------------------------------------
# 2. ANALISIS  (equivalente al Audio Spectrum / Audio Analyze CHOP)
# --------------------------------------------------------------------------------------
@dataclass
class Analysis:
    sr: int
    hop: int
    n_fft: int
    freqs: np.ndarray            # (n_bins,)
    times: np.ndarray            # (n_frames,)
    mag: np.ndarray              # (n_bins, n_frames) magnitud lineal
    db: np.ndarray               # (n_bins, n_frames) dB
    centroid: np.ndarray
    spread: np.ndarray
    flux: np.ndarray
    tonality: np.ndarray         # flatness en dB (mas negativo = mas tonal)
    rms: np.ndarray
    peaks: list = field(default_factory=list)   # por cuadro: (freq[], db[], amp[])
    ref_db: float = 0.0                          # pico global del archivo (referencia)


def stft_mag(x: np.ndarray, n_fft: int, hop: int, chunk: int = 2048) -> np.ndarray:
    """Magnitud del STFT, calculada POR BLOQUES.

    La version ingenua (armar todos los cuadros en un solo array) explota con
    archivos largos: 1 hora de audio con hop=256 son 716k cuadros x 2048 muestras
    = 5.5 GB solo para la matriz temporal. Asi el pico de memoria es de ~16 MB.
    """
    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect" if len(x) > pad else "constant")
    n_frames = max(1 + (len(xp) - n_fft) // hop, 0)
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for s0 in range(0, n_frames, chunk):
        s1 = min(s0 + chunk, n_frames)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(s0, s1)[:, None]
        fr = xp[idx] * win
        out[:, s0:s1] = np.abs(np.fft.rfft(fr, axis=1)).T
    return out


def fit_memory(n_samples: int, n_fft: int, hop: int, limit_mb: float = 800.0) -> int:
    """Sube el hop si el analisis no entraria en memoria. Devuelve el hop a usar."""
    bins = n_fft // 2 + 1
    while (n_samples / hop) * bins * 8 / 1e6 > limit_mb and hop < n_fft:
        hop *= 2
    return hop


def analyze(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 256,
            fmin: float = 300.0, fmax: float = 12000.0,
            peak_floor_db: float = 14.0, max_peaks: int = 10,
            gate_db: float = 45.0) -> Analysis:
    hop_ok = fit_memory(len(x), n_fft, hop)
    if hop_ok != hop:
        print(f"[mem] audio muy largo: subo hop {hop} -> {hop_ok} para que entre en RAM. "
              f"Analiza un tramo mas corto si queres mas resolucion temporal.",
              file=sys.stderr)
        hop = hop_ok
    mag = stft_mag(x, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = np.arange(mag.shape[1]) * hop / sr
    db = 20.0 * np.log10(mag + 1e-9)

    # ---- descriptores por cuadro (los que salen en el HUD del video original)
    p = mag / (mag.sum(axis=0, keepdims=True) + 1e-12)          # distrib. de prob.
    centroid = (freqs[:, None] * p).sum(axis=0)
    spread = np.sqrt(((freqs[:, None] - centroid[None, :]) ** 2 * p).sum(axis=0))
    flux = np.concatenate([[0.0], np.sqrt(((np.diff(mag, axis=1).clip(min=0)) ** 2).sum(axis=0))])
    geo = np.exp(np.log(mag + 1e-9).mean(axis=0))
    ari = mag.mean(axis=0) + 1e-9
    tonality = 10.0 * np.log10(geo / ari)                        # spectral flatness (dB)
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1)) \
        if len(x) >= hop else np.zeros(1)
    rms = np.interp(np.linspace(0, len(rms) - 1, len(times)), np.arange(len(rms)), rms)

    # ---- extraccion de picos espectrales (los NODOS del grafo)
    band = (freqs >= fmin) & (freqs <= fmax)
    bidx = np.where(band)[0]
    ref_db = float(db[bidx].max())
    peaks = []
    for j in range(db.shape[1]):
        col = db[bidx, j]
        if col.max() < ref_db - gate_db:          # gate absoluto: silencio => sin nodos
            peaks.append((np.zeros(0, np.float32),) * 3)
            continue
        floor = max(np.percentile(col, 60) + peak_floor_db, ref_db - gate_db)
        pk, props = find_peaks(col, height=floor, distance=2, prominence=6.0)
        if len(pk) > max_peaks:                                  # quedarse con los mas fuertes
            keep = np.argsort(props["peak_heights"])[-max_peaks:]
            pk = np.sort(pk[keep])
        f_ref, d_ref = [], []
        for k in pk:                                             # interpolacion parabolica
            if 0 < k < len(col) - 1:
                a, b, c = col[k - 1], col[k], col[k + 1]
                dn = a - 2 * b + c
                delta = 0.5 * (a - c) / dn if abs(dn) > 1e-9 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
            else:
                delta = 0.0
            bin_i = bidx[k] + delta
            f_ref.append(bin_i * sr / n_fft)
            d_ref.append(float(col[k]))
        f_ref = np.array(f_ref, dtype=np.float32)
        d_ref = np.array(d_ref, dtype=np.float32)
        peaks.append((f_ref, d_ref, 10.0 ** (d_ref / 20.0)))

    return Analysis(sr, hop, n_fft, freqs, times, mag, db,
                    centroid, spread, flux, tonality, rms, peaks, ref_db)


# --------------------------------------------------------------------------------------
# 3. MODELO / GRAFO  (esto es lo que reemplaza al "particle system")
# --------------------------------------------------------------------------------------
@dataclass
class Graph:
    xy: np.ndarray               # (n,2) posiciones normalizadas 0..1
    t: np.ndarray                # tiempo de cada nodo
    f: np.ndarray                # frecuencia
    a: np.ndarray                # amplitud normalizada 0..1
    frame: np.ndarray            # indice de cuadro
    edges_partial: list          # [(i,j,w)]
    edges_harm: list
    edges_web: list
    hot: int                     # nodo dominante


def build_graph(an: Analysis, i0: int, i1: int,
               fmin: float = 300.0, fmax: float = 12000.0,
               semitone_tol: float = 1.2, knn: int = 3,
               dyn_range: float = 42.0) -> Graph | None:
    """Construye el grafo para la ventana de cuadros [i0, i1)."""
    T, F, D, FR = [], [], [], []
    for j in range(i0, i1):
        f, d, _ = an.peaks[j]
        for k in range(len(f)):
            T.append(an.times[j]); F.append(f[k]); D.append(d[k]); FR.append(j)
    if len(T) < 4:
        return None

    t = np.array(T); f = np.array(F); d = np.array(D); fr = np.array(FR)

    # amplitud: referida al pico GLOBAL del archivo para que los pasajes suaves
    # no se vean igual de brillantes que los fuertes (pero sin apagarse del todo)
    dmax = max(float(d.max()), an.ref_db - 20.0)
    a = np.clip((d - (dmax - dyn_range)) / dyn_range, 0.02, 1.0)

    # posicion: x = tiempo, y = log-frecuencia
    t0, t1 = an.times[i0], an.times[max(i1 - 1, i0)] + 1e-6
    x = (t - t0) / (t1 - t0)
    y = (np.log2(np.clip(f, fmin, fmax)) - np.log2(fmin)) / (np.log2(fmax) - np.log2(fmin))
    xy = np.stack([x, y], axis=1)

    # --- (a) cadenas de parciales: unir picos de cuadros consecutivos con freq similar
    edges_partial = []
    by_frame: dict[int, list[int]] = {}
    for i, jf in enumerate(fr):
        by_frame.setdefault(int(jf), []).append(i)
    tol = semitone_tol / 12.0
    for j in range(i0, i1 - 1):
        A, B = by_frame.get(j, []), by_frame.get(j + 1, [])
        if not A or not B:
            continue
        cost = np.abs(np.log2(f[A][:, None] / f[B][None, :]))
        used = set()
        for ai in np.argsort(-a[A]):                     # los mas fuertes eligen primero
            bi = int(np.argmin(cost[ai]))
            if cost[ai, bi] < tol and bi not in used:
                used.add(bi)
                edges_partial.append((A[ai], B[bi], float(min(a[A[ai]], a[B[bi]]))))

    # --- (b) abanico armonico desde el nodo dominante
    hot = int(np.argmax(d))
    f0 = f[hot]
    edges_harm = []
    same = np.where(fr == fr[hot])[0]
    for i in same:
        if i == hot:
            continue
        r = f[i] / f0
        n = round(r)
        if n >= 1 and abs(r - n) < 0.06:                 # +-6% => parcial armonico n
            edges_harm.append((hot, int(i), float(a[i])))

    # --- (c) MST sobre kNN: los hilos blancos que amarran toda la estructura
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(d2)
    n = len(t)
    k = min(knn, n - 1)
    nn = np.argsort(dist, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k); cols = nn.ravel()
    W = coo_matrix((dist[rows, cols] + 1e-6, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(W.tocsr()).tocoo()
    seen = {(min(i, j), max(i, j)) for i, j, _ in edges_partial}
    edges_web = [(int(i), int(j), float(w)) for i, j, w in zip(mst.row, mst.col, mst.data)
                 if (min(i, j), max(i, j)) not in seen]

    return Graph(xy, t, f, a, fr, edges_partial, edges_harm, edges_web, hot)


# --------------------------------------------------------------------------------------
# 4. RENDER  (glow falso por capas: el truco para el look "fosforescente")
# --------------------------------------------------------------------------------------
GLOW_W = (7.0, 3.2, 1.6, 0.8)
GLOW_A = (0.035, 0.08, 0.18, 0.95)


def _glow_edges(ax, segs, colors, widths, alphas, zorder=2,
                layers=GLOW_W, layer_alphas=GLOW_A):
    """Dibuja un set de aristas con 'bloom' falso: 4 capas de grosor decreciente.
    Vectorizado con LineCollection (clave para que el video no tarde una eternidad)."""
    from matplotlib.collections import LineCollection
    if len(segs) == 0:
        return
    segs = np.asarray(segs)
    widths = np.asarray(widths, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    rgba = np.array([to_rgba(c) for c in colors])
    for lw, la in zip(layers, layer_alphas):
        col = rgba.copy()
        col[:, 3] = np.clip(alphas * la, 0, 1)
        lc = LineCollection(segs, colors=col, linewidths=widths * lw,
                            capstyle="round", zorder=zorder)
        ax.add_collection(lc)


def hz_to_note(fhz: float) -> str:
    if fhz <= 0 or not np.isfinite(fhz):
        return "--"
    m = 69 + 12 * np.log2(fhz / 440.0)
    m = int(round(m))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def render_frame(an: Analysis, g: Graph | None, j_now: int, fig, ax_main, ax_spec, ax_hud,
                 fmin: float, fmax: float, label_top: int = 8, title: str = ""):
    for ax in (ax_main, ax_spec, ax_hud):
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax_main.set_xlim(-0.06, 1.06); ax_main.set_ylim(-0.05, 1.05)

    # ---------- tira de espectro vertical (izquierda)
    col = an.db[:, j_now]
    band = (an.freqs >= fmin) & (an.freqs <= fmax)
    fv = an.freqs[band]
    cv = np.clip((col[band] - (col.max() - 55)) / 55.0, 0, 1)
    yv = (np.log2(fv) - np.log2(fmin)) / (np.log2(fmax) - np.log2(fmin))
    ax_spec.set_xlim(0, 1); ax_spec.set_ylim(-0.05, 1.05)
    ax_spec.hlines(yv, 0.0, cv * 0.95, colors=[(0.15 + 0.85 * v, 0.25 * v + 0.1, 0.55 + 0.45 * (1 - v), min(1, 0.15 + v))
                                               for v in cv], lw=1.0)

    # ---------- eje de frecuencia (derecha)
    for fref in (1000, 2000, 4000, 8000, 12000):
        if fmin <= fref <= fmax:
            yy = (np.log2(fref) - np.log2(fmin)) / (np.log2(fmax) - np.log2(fmin))
            ax_main.plot([0.995, 1.02], [yy, yy], color=C_TXT, lw=0.6, alpha=0.5,
                         transform=ax_main.transAxes, clip_on=False)
            ax_main.text(1.028, yy, f"{fref//1000} kHz" if fref >= 1000 else str(fref),
                         color=C_TXT, fontsize=5.2, va="center", ha="left",
                         transform=ax_main.transAxes, family="monospace")

    if g is not None:
        xy, a = g.xy, g.a
        seg = lambda es: np.array([[xy[i], xy[j]] for i, j, _ in es]) if es else np.empty((0, 2, 2))

        # (c) hilos finos del MST: una sola coleccion, sin glow
        if g.edges_web:
            from matplotlib.collections import LineCollection
            ax_main.add_collection(LineCollection(seg(g.edges_web), colors=to_rgba(C_WEB, 0.16),
                                                  linewidths=0.35, zorder=1))
        # (a) cadenas de parciales
        if g.edges_partial:
            w = np.array([e[2] for e in g.edges_partial])
            _glow_edges(ax_main, seg(g.edges_partial),
                        [C_PARTIAL if v > 0.35 else C_PARTIAL2 for v in w],
                        0.35 + 1.1 * w, 0.35 + 0.65 * w, zorder=3)
        # (b) abanico armonico
        if g.edges_harm:
            w = np.array([e[2] for e in g.edges_harm])
            _glow_edges(ax_main, seg(g.edges_harm), [C_HARM] * len(w),
                        0.4 + 0.9 * w, 0.4 + 0.6 * w, zorder=4,
                        layers=(9.0, 4.0, 1.8, 0.9), layer_alphas=(0.05, 0.1, 0.22, 0.95))

        # ---------- nodos (vectorizado: 3 scatter en total)
        deg = np.zeros(len(a))
        for i, j, _ in g.edges_partial + g.edges_web + g.edges_harm:
            deg[i] += 1; deg[j] += 1
        cols = np.array([to_rgba(C_PARTIAL)] * len(a))
        size = 30 * a + 4
        m = a > 0.6
        cols[m] = to_rgba(C_HOT); size[m] = 46 * a[m] + 8
        m = deg <= 1
        cols[m] = to_rgba(C_ISO); size[m] = 26 * a[m] + 5
        cols[g.hot] = to_rgba(C_HOT); size[g.hot] = 90 * a[g.hot] + 26
        ax_main.scatter(xy[:, 0], xy[:, 1], s=size * 4.2, c=cols, alpha=0.055,
                        zorder=5, linewidths=0)
        ax_main.scatter(xy[:, 0], xy[:, 1], s=size, c=cols, alpha=0.9,
                        zorder=6, linewidths=0)
        ax_main.scatter(xy[:, 0], xy[:, 1], s=size * 2.4, c=cols, alpha=0.5,
                        zorder=6, marker="+", linewidths=0.35)

        # etiquetas numericas de los picos mas fuertes (como en la referencia)
        for i in np.argsort(-a)[:label_top]:
            ax_main.text(xy[i, 0] + 0.012, xy[i, 1] + 0.012,
                         f"{g.f[i]/1000:.2f}k\n{a[i]:.2f}",
                         color=C_HUD, fontsize=4.1, alpha=0.75, family="monospace",
                         linespacing=1.1, zorder=7)

    # ---------- HUD de descriptores
    ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)
    cen, spr = an.centroid[j_now], an.spread[j_now]
    flx = an.flux[j_now] / (an.flux.max() + 1e-9)
    ton = an.tonality[j_now]
    rows = [("SPECTRAL FLUX", f"{flx:0.4f}", flx),
            ("SPECTRAL SPREAD (kHz)", f"{spr/1000:0.2f}", min(spr / 6000, 1)),
            ("SPECTRAL CENTROID (piano)", f"{hz_to_note(cen)}  {cen/1000:0.2f}k", min(cen / 8000, 1)),
            ("TONALITY (dB)", f"{ton:0.1f}", float(np.clip((ton + 30) / 30, 0, 1)))]
    for r, (name, val, lvl) in enumerate(rows):
        y = 0.86 - r * 0.26
        ax_hud.text(1.0, y, name, color=C_TXT, fontsize=4.6, ha="right", va="center",
                    family="monospace")
        ax_hud.text(1.0, y - 0.1, val, color=C_HUD, fontsize=5.4, ha="right", va="center",
                    family="monospace")
        ax_hud.scatter(0.06, y - 0.03, s=34 + 90 * lvl, color=C_ACC, alpha=0.12, linewidths=0)
        ax_hud.scatter(0.06, y - 0.03, s=8 + 26 * lvl, color=C_ACC, alpha=0.9, linewidths=0)

    if title:
        ax_main.text(0.0, 1.035, title, color=C_TXT, fontsize=5.0, family="monospace",
                     transform=ax_main.transAxes, alpha=0.8)
    ax_main.text(0.0, -0.035, f"t = {an.times[j_now]:0.3f} s", color=C_TXT, fontsize=5.0,
                 family="monospace", transform=ax_main.transAxes, alpha=0.8)


def make_figure(w_px=1080, h_px=1350, dpi=200):
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor=BG)
    ax_main = fig.add_axes([0.10, 0.055, 0.80, 0.925])
    ax_spec = fig.add_axes([0.028, 0.055, 0.055, 0.925])
    ax_hud  = fig.add_axes([0.60, 0.012, 0.34, 0.075])
    return fig, ax_main, ax_spec, ax_hud


# --------------------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------------------
def export_csv(an: Analysis, out: str):
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "rms", "centroid_hz", "spread_hz", "flux", "tonality_db",
                    "n_peaks", "f0_hz", "peak_freqs_hz", "peak_db"])
        for j, tt in enumerate(an.times):
            f, d, _ = an.peaks[j]
            f0 = float(f[np.argmax(d)]) if len(f) else 0.0
            w.writerow([f"{tt:.5f}", f"{an.rms[j]:.6f}", f"{an.centroid[j]:.2f}",
                        f"{an.spread[j]:.2f}", f"{an.flux[j]:.5f}", f"{an.tonality[j]:.3f}",
                        len(f), f"{f0:.2f}",
                        ";".join(f"{v:.1f}" for v in f), ";".join(f"{v:.1f}" for v in d)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualizador de grafos espectrales de canto de aves")
    ap.add_argument("input", help="wav/mp3/flac/mp4/mkv/mov")
    ap.add_argument("--mode", choices=["still", "video", "csv"], default="still")
    ap.add_argument("--out", default=None)
    ap.add_argument("--at", type=float, default=None, help="segundo a renderizar (still)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window", type=float, default=0.9, help="ventana temporal visible (s)")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--fmin", type=float, default=300.0)
    ap.add_argument("--fmax", type=float, default=12000.0)
    ap.add_argument("--max-peaks", type=int, default=10)
    ap.add_argument("--floor-db", type=float, default=14.0)
    ap.add_argument("--gate-db", type=float, default=45.0,
                    help="dB bajo el pico global a partir de los cuales se considera silencio")
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-audio", action="store_true", help="no pegar el audio al mp4")
    args = ap.parse_args(argv)

    x, sr = load_audio(args.input, args.sr)
    if args.end is not None:
        x = x[: int(args.end * sr)]
    if args.start:
        x = x[int(args.start * sr):]
    print(f"[audio] {len(x)/sr:0.2f} s @ {sr} Hz", file=sys.stderr)

    an = analyze(x, sr, args.n_fft, args.hop, args.fmin, args.fmax,
                 args.floor_db, args.max_peaks, args.gate_db)
    print(f"[stft] {an.db.shape[1]} cuadros, {sum(len(p[0]) for p in an.peaks)} picos",
          file=sys.stderr)

    base = os.path.splitext(os.path.basename(args.input))[0]

    if args.mode == "csv":
        out = args.out or f"{base}_features.csv"
        export_csv(an, out)
        print(f"[ok] {out}")
        return

    global plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F811

    win_frames = max(4, int(args.window * sr / args.hop))
    fig, ax_main, ax_spec, ax_hud = make_figure(args.width, args.height, args.dpi)
    title = f"{base}  |  n_fft={args.n_fft} hop={args.hop} sr={sr}"

    if args.mode == "still":
        if args.at is None:
            j = int(np.argmax(an.rms))                     # el momento mas fuerte
        else:
            j = int(np.clip(args.at * sr / args.hop, 0, an.db.shape[1] - 1))
        i0 = max(0, j - win_frames // 2); i1 = min(an.db.shape[1], i0 + win_frames)
        g = build_graph(an, i0, i1, args.fmin, args.fmax, knn=args.knn)
        render_frame(an, g, j, fig, ax_main, ax_spec, ax_hud, args.fmin, args.fmax, title=title)
        out = args.out or f"{base}_graph.png"
        fig.savefig(out, facecolor=BG)
        print(f"[ok] {out}")
        return

    # ---- video
    out = args.out or f"{base}_graph.mp4"
    tmpdir = tempfile.mkdtemp(prefix="birdgraph_")
    n_out = int(len(x) / sr * args.fps)
    for n in range(n_out):
        tsec = n / args.fps
        j = int(np.clip(tsec * sr / args.hop, 0, an.db.shape[1] - 1))
        i0 = max(0, j - win_frames + 1); i1 = j + 1                  # ventana que "avanza"
        g = build_graph(an, i0, i1, args.fmin, args.fmax, knn=args.knn)
        render_frame(an, g, j, fig, ax_main, ax_spec, ax_hud, args.fmin, args.fmax, title=title)
        fig.savefig(os.path.join(tmpdir, f"f{n:06d}.png"), facecolor=BG)
        if n % 15 == 0:
            print(f"\r[render] {n+1}/{n_out}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(args.fps),
           "-i", os.path.join(tmpdir, "f%06d.png")]
    if not args.no_audio:
        cmd += ["-ss", str(args.start), "-i", args.input, "-map", "0:v", "-map", "1:a?",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[ok] {out}")


if __name__ == "__main__":
    if len(sys.argv) == 1 and INPUT:
        print(f"[config] usando INPUT del archivo: {INPUT}", file=sys.stderr)
        main([INPUT] + DEFAULTS)
    else:
        main()
