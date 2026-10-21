#!/usr/bin/env python3
"""
birdgraph_live.py -- Video + grafo espectral 3D en vivo, sincronizados.

NO NECESITA TERMINAL: pone la ruta en VIDEO aca abajo y dale Run.
Si dejas VIDEO = "" te abre un selector de archivos.

Analisis (todo calculado a mano con numpy/scipy, sin librosa):
  - STFT -> picos espectrales (parciales) con interpolacion parabolica
  - MFCC 40-dim + deltas -> envolvente timbrica y dinamica espectral
  - PCA -> proyeccion a 3D: trayectorias donde la cercania = similitud timbrica
  - Chroma (12 clases de altura) + concentracion cromatica
  - Descriptores: centroide, spread, flux, tonalidad (flatness)
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

# =====================================================================================
#  CONFIG   <<<<<<<<<<  EDITA SOLO ESTO Y DALE RUN
# =====================================================================================

VIDEO = r""          # ej: r"C:\ruta\a\tu\pajaro.mp4"
                     # vacio = te abre un selector de archivos

LAYOUT = "manifold"  # manifold | circle | helix | wall | sphere   (teclas 1..5)

# --- banda de analisis (ajustar por especie: koel 500-5000, jilguero 1500-9000)
FMIN = 400.0
FMAX = 9000.0

# --- que tramo del archivo analizar (IMPORTANTE con videos largos)
START_AT = 0.0       # segundo desde donde empezar
DURATION = 180.0     # cuantos segundos analizar. 0 = todo (ojo con la RAM:
                     # 1 hora con HOP=256 pide ~6 GB, se sube el hop solo)

# --- ventanas de tiempo
WINDOW = 1.2         # segundos visibles en los layouts de picos
TRAIL = 6.0          # segundos de estela en el layout manifold

# --- densidad / rendimiento
NODE_HOP = 3         # 1 = denso y lento, 4-6 = liviano
MAX_PEAKS = 7        # picos espectrales por cuadro
KNN = 3              # densidad de la telarana blanca
UI_FPS = 60

# --- STFT
N_FFT = 2048
HOP = 256
SR = 44100
FLOOR_DB = 14.0      # umbral sobre el piso de ruido para aceptar un pico
GATE_DB = 45.0       # cuanto por debajo del pico global es silencio

# --- MFCC / PCA
N_MFCC = 40
N_MELS = 64
USE_DELTAS = True    # agrega la derivada temporal (dinamica espectral)
SMOOTH = 5           # suavizado de la trayectoria PCA (cuadros)

# --- desvanecido (que el rastro se borre solo, sin acumular ruido)
FADE_HALFLIFE = 0.45      # segundos: cada 0.45 s el nodo pierde la mitad del brillo
FADE_HALFLIFE_MANIFOLD = 1.6   # en manifold conviene mas largo (mostrar estructura)
ALPHA_CUTOFF = 0.035      # por debajo de esto el nodo se descarta (limpia y acelera)
SHRINK_WITH_FADE = True   # ademas de apagarse, el nodo se encoge

# --- etiquetas de datos sobre los nodos
SHOW_LABELS = True
MAX_LABELS = 9            # cuantas etiquetas simultaneas
LABEL_MIN_SEP = 0.16      # separacion 3D minima entre etiquetas (evita amontonarse)

# --- estetica
SHOW_VIDEO = True
AUTOROTATE = False
COLOR_BY_CHROMA = True   # en manifold, colorea por clase de altura dominante
FLUX_PULSE = 0.18        # cuanto pulsa la escala con el spectral flux (0 = nada)
TOL_SEMITONES = 2.5      # tolerancia para unir parciales entre cuadros

# =====================================================================================

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    from birdgraph import load_audio, analyze, hz_to_note
    from birdfeatures import mfcc, chroma, chroma_concentration, PCA3, smooth_traj
except ImportError as e:
    sys.exit(f"Falta un archivo en la misma carpeta: {e}\n"
             "Necesito birdgraph.py y birdfeatures.py junto a este script.")

# cv2 se importa ANTES que Qt y hay que sacarle su plugin path: opencv-python trae
# sus propias DLL de Qt y rompe PyQt5. Con opencv-python-headless no pasa.
try:
    import cv2
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
except ImportError:
    cv2 = None

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    import sounddevice as sd
except Exception:
    sd = None

# --------------------------------------------------------------------------- paleta
C_WEB      = (0.72, 0.80, 0.90)
C_PARTIAL  = (1.00, 0.18, 0.48)
C_HARM     = (0.14, 1.00, 0.60)
C_HOT      = (1.00, 1.00, 1.00)
C_ISO      = (0.95, 1.00, 0.27)
CSS_HUD    = "color:#9fb3c8; font-family:Consolas,monospace; font-size:11px;"
CSS_VAL    = "color:#e9f2fb; font-family:Consolas,monospace; font-size:15px;"
LAYOUTS    = ["manifold", "circle", "helix", "wall", "sphere"]


def chroma_rgb(pc: np.ndarray) -> np.ndarray:
    """Clase de altura (0..11) -> RGB por circulo de quintas (colores vecinos = notas
    armonicamente cercanas, no cromaticamente cercanas)."""
    fifths = (pc * 7) % 12
    h = fifths / 12.0
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    v, s = 1.0, 0.78
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    o = np.ones_like(f) * v
    r = np.choose(i, [o, q, p * np.ones_like(f), p * np.ones_like(f), t, o])
    g = np.choose(i, [t, o, o, q, p * np.ones_like(f), p * np.ones_like(f)])
    b = np.choose(i, [p * np.ones_like(f), p * np.ones_like(f), t, o, o, q])
    return np.stack([r, g, b], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- reloj
class Clock:
    """Reloj maestro. Si hay sounddevice lo manda el audio (posicion real de muestra);
    si no, cae a reloj interno y sigue andando sin sonido."""

    def __init__(self, x, sr, block=1024):
        self.x, self.sr = x, sr
        self.pos, self.playing = 0, False
        self.lock = threading.Lock()
        self._t0 = 0.0
        self.stream = None
        if sd is not None:
            try:
                self.stream = sd.OutputStream(samplerate=sr, channels=1, blocksize=block,
                                              dtype="float32", callback=self._cb)
                self.stream.start()
            except Exception as e:
                print(f"[audio] sin salida ({e}); sincronizo por reloj interno",
                      file=sys.stderr)

    def _cb(self, out, frames, t, status):
        with self.lock:
            if not self.playing:
                out[:] = 0.0
                return
            end = min(self.pos + frames, len(self.x))
            n = end - self.pos
            out[:n, 0] = self.x[self.pos:end]
            if n < frames:
                out[n:, 0] = 0.0
                self.playing = False
            self.pos = end

    def play(self):
        with self.lock:
            if self.pos >= len(self.x):
                self.pos = 0
            self.playing, self._t0 = True, time.perf_counter()

    def pause(self):
        with self.lock:
            if self.stream is None and self.playing:
                self.pos += int((time.perf_counter() - self._t0) * self.sr)
            self.playing = False

    def toggle(self):
        self.pause() if self.playing else self.play()

    def seek(self, seconds):
        with self.lock:
            self.pos = int(np.clip(seconds * self.sr, 0, max(len(self.x) - 1, 0)))
            self._t0 = time.perf_counter()

    @property
    def t(self):
        with self.lock:
            p = self.pos
            if self.stream is None and self.playing:
                p += int((time.perf_counter() - self._t0) * self.sr)
            return min(p, len(self.x)) / self.sr

    def close(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close()


# ------------------------------------------------------------------ features extra
class Features:
    """MFCC -> PCA 3D + chroma. Se calcula una sola vez al abrir el archivo."""

    def __init__(self, an, sr):
        M = mfcc(an.mag, sr, N_FFT, N_MFCC, N_MELS, FMIN, FMAX, deltas=USE_DELTAS)
        # se descarta c0 (energia global): queremos timbre, no volumen
        M = np.delete(M, [0, N_MFCC] if USE_DELTAS else [0], axis=1)
        self.active = an.db.max(axis=0) > an.ref_db - GATE_DB
        self.pca = PCA3(M, self.active)
        Y = smooth_traj(self.pca.transform(M), SMOOTH)
        r = np.percentile(np.abs(Y[self.active]) if self.active.any() else np.abs(Y), 97)
        self.Y = (Y / max(r, 1e-6)).astype(np.float32)        # normalizado a ~[-1,1]
        self.explained = self.pca.explained
        self.chroma = chroma(an.mag, sr, N_FFT, FMIN, FMAX)
        self.conc = chroma_concentration(self.chroma)
        self.pc = self.chroma.argmax(axis=0)                  # clase dominante
        self.rgb = chroma_rgb(self.pc)


# ------------------------------------------------------------------------- grafos
class Builder:
    def __init__(self, an, feat):
        self.an, self.feat = an, feat
        self.tol = TOL_SEMITONES / 12.0

    # ---- layouts basados en PICOS espectrales
    def peaks_graph(self, j_now, win_frames, layout, flux):
        an = self.an
        # el desvanecido exponencial define hasta donde vale la pena mirar atras:
        # mas alla del cutoff el nodo ya es invisible, asi que ni se construye
        max_age = FADE_HALFLIFE * np.log2(1.0 / ALPHA_CUTOFF)
        win_frames = min(win_frames, max(6, int(max_age * an.sr / an.hop)))
        i0 = max(0, j_now - win_frames + 1)
        idxs = list(range(i0, j_now + 1, NODE_HOP))
        slot, F, D = [], [], []
        for s, j in enumerate(idxs):
            f, d, _ = an.peaks[j]
            slot += [s] * len(f); F.append(f); D.append(d)
        if len(slot) < 4:
            return None
        slot = np.array(slot, np.int32)
        f = np.concatenate(F) if F else np.zeros(0)
        d = np.concatenate(D) if D else np.zeros(0)
        n_slots = max(len(idxs) - 1, 1)
        t_rel = slot / n_slots - 1.0

        dmax = max(float(d.max()), an.ref_db - 20.0)
        a = np.clip((d - (dmax - 42.0)) / 42.0, 0.02, 1.0)

        span = np.log2(FMAX / FMIN)
        oc = np.clip(np.log2(np.clip(f, FMIN, FMAX) / FMIN) / span, 0, 1)
        chrm = (np.log2(np.clip(f, FMIN, FMAX) / FMIN)) % 1.0
        spread_n = float(np.clip(an.spread[j_now] / 4000.0, 0.25, 1.6))

        if layout == "wall":
            x, y, z = t_rel * 2 + 1, oc * 2 - 1, a * 1.3 - 0.65
        elif layout == "sphere":
            az, el = 2 * np.pi * chrm, (oc - 0.5) * np.pi * 0.9
            r = 0.45 + 0.55 * (1 + t_rel) + 0.25 * a
            x, y, z = (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
                       r * np.sin(el))
        elif layout == "circle":
            # "circle acoustic": tiempo -> radio, frecuencia -> angulo (galaxia).
            # el spread espectral abre o cierra el anillo.
            ang = 2 * np.pi * oc
            r = (0.12 + 1.05 * (1 + t_rel)) * spread_n
            x, y, z = r * np.cos(ang), r * np.sin(ang), (a - 0.5) * 0.55
        else:                                                  # helix
            ang = 2 * np.pi * chrm
            r = 0.22 + 0.78 * oc + 0.12 * a
            x, y, z = r * np.cos(ang), r * np.sin(ang), t_rel * 2.7 + 1.35
        pos = np.stack([x, y, z], 1).astype(np.float32) * (1.0 + FLUX_PULSE * flux)

        by_slot = {}
        for i, s in enumerate(slot):
            by_slot.setdefault(int(s), []).append(i)

        e_part, w_part = [], []
        for s in range(len(idxs) - 1):
            A, B = by_slot.get(s, []), by_slot.get(s + 1, [])
            if not A or not B:
                continue
            cost = np.abs(np.log2(f[A][:, None] / f[B][None, :]))
            used = set()
            for ai in np.argsort(-a[A]):
                bi = int(np.argmin(cost[ai]))
                if cost[ai, bi] < self.tol and bi not in used:
                    used.add(bi)
                    e_part.append((A[ai], B[bi])); w_part.append(min(a[A[ai]], a[B[bi]]))

        hot = int(np.argmax(d)); f0 = f[hot]
        e_harm, w_harm = [], []
        for i in by_slot.get(int(slot[hot]), []):
            if i != hot:
                rr = f[i] / f0; nh = round(rr)
                if nh >= 1 and abs(rr - nh) < 0.06:
                    e_harm.append((hot, i)); w_harm.append(a[i])

        e_web = self._mst(np.stack([slot / n_slots, oc], 1), e_part)
        age_s = ((len(idxs) - 1 - slot) * NODE_HOP) * an.hop / an.sr
        fade = np.power(0.5, age_s / max(FADE_HALFLIFE, 1e-3)).astype(np.float32)
        col = np.tile(np.array(C_PARTIAL, np.float32), (len(a), 1))
        return dict(pos=pos, a=a, fade=fade, hot=hot, base_col=col, use_role=True,
                    f=f, d=d, age=age_s,
                    e_part=e_part, w_part=np.array(w_part) if w_part else np.zeros(0),
                    e_harm=e_harm, w_harm=np.array(w_harm) if w_harm else np.zeros(0),
                    e_web=e_web)

    # ---- layout MANIFOLD: nodos = cuadros proyectados por PCA de MFCC
    def manifold_graph(self, j_now, flux):
        an, ft = self.an, self.feat
        i0 = max(0, j_now - int(TRAIL * an.sr / an.hop) + 1)
        idxs = np.arange(i0, j_now + 1, NODE_HOP)
        idxs = idxs[ft.active[idxs]]                   # solo cuadros con senal
        if len(idxs) < 4:
            return None
        age_s = (j_now - idxs) * an.hop / an.sr
        keep_age = age_s <= FADE_HALFLIFE_MANIFOLD * np.log2(1.0 / ALPHA_CUTOFF)
        idxs, age_s = idxs[keep_age], age_s[keep_age]
        if len(idxs) < 4:
            return None
        pos = ft.Y[idxs] * (1.0 + FLUX_PULSE * flux)
        fade = np.power(0.5, age_s / max(FADE_HALFLIFE_MANIFOLD, 1e-3)).astype(np.float32)
        rms = an.rms[idxs]
        a = np.clip(rms / (rms.max() + 1e-9), 0.05, 1.0) * 0.4 + 0.6 * ft.conc[idxs]

        # (1) trayectoria temporal: cuadros consecutivos
        keep = np.diff(idxs) <= NODE_HOP * 3            # no unir a traves de silencios
        e_part = [(i, i + 1) for i in np.where(keep)[0]]
        w_part = np.minimum(a[:-1], a[1:])[keep] if len(a) > 1 else np.zeros(0)

        # (2) vecinos en el manifold: cercania = similitud timbrica, aunque esten
        #     lejos en el tiempo. Estas son las cuerdas que "cosen" el ovillo.
        tree = cKDTree(pos)
        k = min(3, len(pos))
        dist, nn = tree.query(pos, k=k)
        e_harm, w_harm = [], []
        thr = np.percentile(dist[:, 1:], 45) if k > 1 else 0
        for i in range(len(pos)):
            for m in range(1, k):
                j2 = int(nn[i, m])
                if abs(j2 - i) > 6 and dist[i, m] < thr:
                    e_harm.append((i, j2)); w_harm.append(1.0 - dist[i, m] / (thr + 1e-9))

        e_web = self._mst(pos, e_part)
        base = ft.rgb[idxs] if COLOR_BY_CHROMA else np.tile(np.array(C_PARTIAL, np.float32),
                                                            (len(idxs), 1))
        return dict(pos=pos.astype(np.float32), a=a, fade=fade, hot=len(idxs) - 1,
                    base_col=base, use_role=not COLOR_BY_CHROMA,
                    frames=idxs, age=age_s,
                    e_part=e_part, w_part=np.asarray(w_part),
                    e_harm=e_harm, w_harm=np.array(w_harm) if w_harm else np.zeros(0),
                    e_web=e_web)

    def _mst(self, pts, exclude):
        k = min(KNN + 1, len(pts))
        if k < 2:
            return []
        dist, nn = cKDTree(pts).query(pts, k=k)
        rows = np.repeat(np.arange(len(pts)), k - 1)
        W = coo_matrix((dist[:, 1:].ravel() + 1e-6, (rows, nn[:, 1:].ravel())),
                       shape=(len(pts),) * 2)
        mst = minimum_spanning_tree(W.tocsr()).tocoo()
        seen = {(min(i, j), max(i, j)) for i, j in exclude}
        return [(int(i), int(j)) for i, j in zip(mst.row, mst.col)
                if (min(i, j), max(i, j)) not in seen]


def seg_arrays(pos, fade, edges, rgb, weights=None, amin=0.10):
    """Aristas -> formato de GLLinePlotItem(mode='lines'). rgb puede ser un color
    unico o un array (n_nodos,3) para colorear por nodo."""
    if not edges:
        return np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)
    e = np.asarray(edges)
    v = pos[e.ravel()]
    w = np.ones(len(e)) if weights is None or not len(weights) else np.clip(weights, 0, 1)
    al = np.clip(amin + (1 - amin) * w, 0, 1) * np.minimum(fade[e[:, 0]], fade[e[:, 1]])
    c = np.empty((len(e) * 2, 4), np.float32)
    c[:, :3] = np.asarray(rgb)[e.ravel()] if np.ndim(rgb) == 2 else rgb
    c[:, 3] = np.repeat(al, 2)
    return v.astype(np.float32), c


# ------------------------------------------------------------------------ ventana
class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, path, x, sr, an, feat, t_offset=0.0):
        super().__init__()
        self.an, self.sr, self.feat = an, sr, feat
        self.t_offset = t_offset          # el video arranca donde corta START_AT
        self.layout_mode = LAYOUT if LAYOUT in LAYOUTS else "manifold"
        self.window_s, self.autorotate = WINDOW, AUTOROTATE
        self.j_cache = -10**9
        self.frames_drawn, self.fps_t0, self.fps = 0, time.perf_counter(), 0.0

        self.setWindowTitle(f"birdgraph live — {os.path.basename(path)}")
        self.setStyleSheet("background:#08080a;")

        # ---------- video
        self.cap, self.video_fps, self.v_idx = None, 30.0, -1
        if cv2 is not None and SHOW_VIDEO:
            cap = cv2.VideoCapture(path)
            if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
                self.cap = cap
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000;")
        self.video_label.setMinimumWidth(300)
        if self.cap is None:
            self.video_label.setText("sin pista de video")
            self.video_label.setStyleSheet("background:#000;color:#3f4a56;" + CSS_HUD)

        # ---------- vista 3D
        self.view = gl.GLViewWidget()
        try:
            self.view.setBackgroundColor(QtGui.QColor(8, 8, 10))
        except Exception:
            pass
        self.view.setCameraPosition(distance=4.4, elevation=16, azimuth=-58)
        self.it_web = self._line(1.0)
        self.it_part = [self._line(w) for w in (7.0, 3.0, 1.4)]
        self.it_harm = [self._line(w) for w in (9.0, 4.0, 1.7)]
        self.it_halo = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=18, pxMode=True)
        self.it_core = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=5, pxMode=True)
        for it in (self.it_halo, self.it_core):
            it.setGLOptions("additive"); self.view.addItem(it)

        # ---------- espectro + chroma
        self.spec = self._plot(); self.spec.setLogMode(x=True, y=False)
        self.spec.setYRange(0, 1); self.spec.setXRange(np.log10(FMIN), np.log10(FMAX))
        self.curve = self.spec.plot(pen=pg.mkPen("#25ff9a", width=1), fillLevel=0,
                                    brush=pg.mkBrush(37, 255, 154, 34))
        self.pk = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 46, 122, 210), pen=None)
        self.spec.addItem(self.pk)

        self.chr = self._plot(); self.chr.setMaximumWidth(210)
        self.chr.setYRange(0, 1); self.chr.setXRange(-0.6, 11.6)
        self.bars = pg.BarGraphItem(x=np.arange(12), height=np.zeros(12), width=0.7,
                                    brush=pg.mkBrush(233, 242, 251, 150), pen=None)
        self.chr.addItem(self.bars)
        self.chr.getAxis("bottom").setTicks(
            [[(i, n) for i, n in enumerate("C C# D D# E F F# G G# A A# B".split())]])

        # ---------- HUD
        self.hud, row = {}, QtWidgets.QHBoxLayout()
        row.setSpacing(16)
        for key, lab in [("t", "TIME"), ("cen", "CENTROID (piano)"), ("spr", "SPREAD kHz"),
                         ("flx", "FLUX"), ("ton", "TONALITY dB"), ("cc", "CHROMA CONC"),
                         ("pca", "PCA xyz"), ("var", "VAR EXPLAINED"),
                         ("nodes", "NODES/EDGES"), ("lay", "LAYOUT"), ("fps", "FPS")]:
            box = QtWidgets.QVBoxLayout(); box.setSpacing(1)
            c = QtWidgets.QLabel(lab); c.setStyleSheet(CSS_HUD)
            v = QtWidgets.QLabel("--"); v.setStyleSheet(CSS_VAL)
            box.addWidget(c); box.addWidget(v); row.addLayout(box); self.hud[key] = v
        row.addStretch(1)
        self.hud["var"].setText("".join(f"{p*100:.0f}% " for p in feat.explained))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, max(int(len(x) / sr * 100), 1))
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:3px;background:#1d2530;}"
            "QSlider::sub-page:horizontal{background:#25ff9a;}"
            "QSlider::handle:horizontal{width:9px;background:#e9f2fb;margin:-4px 0;"
            "border-radius:4px;}")
        self.slider.sliderMoved.connect(lambda v: self.clock.seek(v / 100.0))

        # ---------- armado
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.split.addWidget(self.video_label); self.split.addWidget(self.view)
        self.split.setSizes([420, 980]); self.split.setHandleWidth(1)

        strip = QtWidgets.QHBoxLayout(); strip.setSpacing(10)
        strip.addWidget(self.spec, 1); strip.addWidget(self.chr)
        lay = QtWidgets.QVBoxLayout()
        lay.setContentsMargins(14, 8, 14, 10); lay.setSpacing(8)
        lay.addWidget(self.split, 1); lay.addLayout(strip)
        lay.addWidget(self.slider); lay.addLayout(row)
        hint = QtWidgets.QLabel("ESPACIO play/pausa   ←→ seek   1..5 layout   [ ] ventana"
                                "   R rotar   V video   F fullscreen   ESC salir")
        hint.setStyleSheet("color:#46525f;font-family:Consolas,monospace;font-size:10px;")
        lay.addWidget(hint)
        w = QtWidgets.QWidget(); w.setLayout(lay); self.setCentralWidget(w)
        self.resize(1460, 900)

        self._make_labels()
        self.builder = Builder(an, feat)
        self.clock = Clock(x, sr)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / UI_FPS))
        self.clock.play()

    # ------------------------------------------------------------ etiquetas
    def _make_labels(self):
        self.label_pool = []
        for _ in range(MAX_LABELS if SHOW_LABELS else 0):
            lb = QtWidgets.QLabel(self.view)
            lb.setStyleSheet(
                "background:rgba(6,9,14,165); border:1px solid rgba(140,170,200,70);"
                "color:#dfe9f3; font-family:Consolas,monospace; font-size:9px;"
                "padding:2px 4px;")
            lb.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            lb.hide()
            self.label_pool.append(lb)
        self.cur_labels = []

    def pick_labels(self, g):
        """Elige los nodos a etiquetar: los mas fuertes, pero separados entre si
        para que las cajas no se amontonen."""
        if not SHOW_LABELS:
            return []
        pos, score = g["pos"], g["a"] * g["fade"]
        order = np.argsort(-score)
        chosen = [int(g["hot"])]
        for i in order:
            if len(chosen) >= MAX_LABELS:
                break
            i = int(i)
            if i in chosen:
                continue
            if min(np.linalg.norm(pos[i] - pos[c]) for c in chosen) > LABEL_MIN_SEP:
                chosen.append(i)
        return chosen

    def label_text(self, g, i, j):
        """Todas las variables que el sistema esta entendiendo en ese punto."""
        an, ft = self.an, self.feat
        hot = i == g["hot"]
        if self.layout_mode == "manifold":
            fr = int(g["frames"][i])
            Y = ft.Y[fr]
            note = "C C# D D# E F F# G G# A A# B".split()[int(ft.pc[fr])]
            L = [f"t {fr*an.hop/an.sr + self.t_offset:6.2f}s",
                 f"PC {Y[0]:+.2f} {Y[1]:+.2f} {Y[2]:+.2f}",
                 f"chr {note:<2} conc {ft.conc[fr]:.2f}"]
            if hot:
                L += [f"cen {an.centroid[fr]/1000:.2f}k  spr {an.spread[fr]/1000:.2f}k",
                      f"flx {an.flux[fr]/(an.flux.max()+1e-9):.3f}  ton {an.tonality[fr]:.1f}",
                      f"rms {an.rms[fr]:.4f}"]
            return "\n".join(L)
        f, d = g["f"][i], g["d"][i]
        L = [f"{f/1000:6.3f} kHz  {hz_to_note(f)}",
             f"{d:6.1f} dB   a {g['a'][i]:.2f}",
             f"age {g['age'][i]*1000:4.0f} ms"]
        if not hot and g["f"][g["hot"]] > 0:
            r = f / g["f"][g["hot"]]
            if abs(r - round(r)) < 0.06 and round(r) >= 1:
                L.append(f"armonico h{int(round(r))}")
        if hot:
            note = "C C# D D# E F F# G G# A A# B".split()[int(self.feat.pc[j])]
            L += [f"cen {an.centroid[j]/1000:.2f}k  spr {an.spread[j]/1000:.2f}k",
                  f"flx {an.flux[j]/(an.flux.max()+1e-9):.3f}  ton {an.tonality[j]:.1f}",
                  f"chr {note:<2} conc {self.feat.conc[j]:.2f}"]
        return "\n".join(L)

    def _mvp(self):
        """Matriz modelo-vista-proyeccion. La firma de projectionMatrix() cambio
        entre versiones de pyqtgraph, asi que pruebo las dos."""
        v = self.view
        try:
            vp = v.getViewport()
            region = (0, 0, vp[2], vp[3])
            return v.projectionMatrix(region, vp) * v.viewMatrix()
        except TypeError:
            try:
                return v.projectionMatrix() * v.viewMatrix()
            except Exception:
                return None
        except Exception:
            return None

    def place_labels(self):
        """Proyecta 3D -> pantalla con la matriz MVP de la vista y mueve las cajas.
        Se hace en cada frame, no solo al reconstruir, asi las etiquetas siguen a la
        camara mientras orbitas."""
        if not SHOW_LABELS:
            return
        mvp = self._mvp()
        if mvp is None:
            return
        W, H = self.view.width(), self.view.height()
        for n, lb in enumerate(self.label_pool):
            if n >= len(self.cur_labels):
                lb.hide(); continue
            p, text, rgb, alpha = self.cur_labels[n]
            v = mvp.map(QtGui.QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0))
            if v.w() <= 1e-6:
                lb.hide(); continue
            x = (v.x() / v.w() * 0.5 + 0.5) * W
            y = (0.5 - v.y() / v.w() * 0.5) * H
            if not (-60 < x < W + 60 and -60 < y < H + 60) or alpha < 0.12:
                lb.hide(); continue
            if lb.text() != text:
                lb.setText(text); lb.adjustSize()
            r, gg, b = (np.clip(np.asarray(rgb), 0, 1) * 255).astype(int)
            lb.setStyleSheet(
                f"background:rgba(6,9,14,{int(60+110*alpha)});"
                f"border:1px solid rgba({r},{gg},{b},{int(40+150*alpha)});"
                f"color:rgba(223,233,243,{alpha:.2f});"
                "font-family:Consolas,monospace; font-size:9px; padding:2px 4px;")
            lb.move(int(x + 10), int(y - lb.height() / 2))
            lb.show()

    def _line(self, width):
        it = gl.GLLinePlotItem(pos=np.zeros((0, 3)), mode="lines", width=width,
                               antialias=True)
        it.setGLOptions("additive"); self.view.addItem(it)
        return it

    def _plot(self):
        p = pg.PlotWidget(background=None)
        p.setMaximumHeight(96); p.setMouseEnabled(False, False); p.hideButtons()
        p.showAxis("left", False)
        p.getAxis("bottom").setPen("#2a3340"); p.getAxis("bottom").setTextPen("#5a6b7d")
        return p

    # ------------------------------------------------------------------ loop
    def tick(self):
        an, sr = self.an, self.sr
        t = self.clock.t
        j = int(np.clip(t * sr / an.hop, 0, an.db.shape[1] - 1))
        flux = float(an.flux[j] / (an.flux.max() + 1e-9))

        if self.autorotate:
            self.view.opts["azimuth"] += 0.12

        if self.cap is not None and self.split.sizes()[0] > 0:
            self._video(t)

        if abs(j - self.j_cache) >= NODE_HOP:
            self.j_cache = j
            if self.layout_mode == "manifold":
                g = self.builder.manifold_graph(j, flux)
            else:
                win = max(6, int(self.window_s * sr / an.hop))
                g = self.builder.peaks_graph(j, win, self.layout_mode, flux)
            self.draw(g)

        self.place_labels()

        band = (an.freqs >= FMIN) & (an.freqs <= FMAX)
        lo = an.ref_db - 60.0
        self.curve.setData(an.freqs[band], np.clip((an.db[band, j] - lo) / 60.0, 0, 1))
        pf, pd, _ = an.peaks[j]
        self.pk.setData(pf, np.clip((pd - lo) / 60.0, 0, 1)) if len(pf) else \
            self.pk.setData([], [])
        ch = self.feat.chroma[:, j]
        self.bars.setOpts(height=ch / (ch.max() + 1e-9))

        self.frames_drawn += 1
        if time.perf_counter() - self.fps_t0 > 0.5:
            self.fps = self.frames_drawn / (time.perf_counter() - self.fps_t0)
            self.frames_drawn, self.fps_t0 = 0, time.perf_counter()
        cen, Y = an.centroid[j], self.feat.Y[j]
        self.hud["t"].setText(f"{t + self.t_offset:6.2f} s")
        self.hud["cen"].setText(f"{hz_to_note(cen)} {cen/1000:5.2f}k")
        self.hud["spr"].setText(f"{an.spread[j]/1000:5.2f}")
        self.hud["flx"].setText(f"{flux:5.3f}")
        self.hud["ton"].setText(f"{an.tonality[j]:6.1f}")
        self.hud["cc"].setText(f"{self.feat.conc[j]:5.3f}")
        self.hud["pca"].setText(f"{Y[0]:+.2f} {Y[1]:+.2f} {Y[2]:+.2f}")
        self.hud["lay"].setText(self.layout_mode)
        self.hud["fps"].setText(f"{self.fps:4.0f}")
        if not self.slider.isSliderDown():
            self.slider.setValue(int(t * 100))
        if not self.clock.playing and t >= len(self.clock.x) / sr - 0.05:
            self.clock.seek(0.0); self.clock.play()

    def _video(self, t):
        target = int((t + self.t_offset) * self.video_fps)
        if target < self.v_idx or target - self.v_idx > 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target); self.v_idx = target - 1
        while self.v_idx < target - 1:
            self.cap.grab(); self.v_idx += 1
        if self.v_idx < target:
            ok, fr = self.cap.read(); self.v_idx += 1
            if ok:
                h, w = fr.shape[:2]
                if hasattr(QtGui.QImage, "Format_BGR888"):
                    img = QtGui.QImage(fr.data, w, h, 3 * w,
                                       QtGui.QImage.Format_BGR888).copy()
                else:
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    img = QtGui.QImage(rgb.data, w, h, 3 * w,
                                       QtGui.QImage.Format_RGB888).copy()
                self.video_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                    self.video_label.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))

    def draw(self, g):
        if g is None:
            for it in [self.it_web, *self.it_part, *self.it_harm]:
                it.setData(pos=np.zeros((2, 3)), color=np.zeros((2, 4)))
            self.it_halo.setData(pos=np.zeros((0, 3)))
            self.it_core.setData(pos=np.zeros((0, 3)))
            self.cur_labels = []
            self.hud["nodes"].setText("0 / 0")
            return
        pos, a, fade, base = g["pos"], g["a"], g["fade"], g["base_col"]

        v, c = seg_arrays(pos, fade, g["e_web"], C_WEB, None, 0.14)
        c[:, 3] *= 0.22
        self.it_web.setData(pos=v, color=c)

        v, c = seg_arrays(pos, fade, g["e_part"], base, g["w_part"], 0.25)
        for it, k in zip(self.it_part, (0.055, 0.13, 0.95)):
            cc = c.copy(); cc[:, 3] *= k; it.setData(pos=v, color=cc)

        v, c = seg_arrays(pos, fade, g["e_harm"], C_HARM, g["w_harm"], 0.30)
        for it, k in zip(self.it_harm, (0.07, 0.16, 0.95)):
            cc = c.copy(); cc[:, 3] *= k; it.setData(pos=v, color=cc)

        col = np.empty((len(a), 4), np.float32)
        col[:, :3] = base
        if g["use_role"]:
            deg = np.zeros(len(a))
            for i, j2 in g["e_part"] + g["e_web"] + g["e_harm"]:
                deg[i] += 1; deg[j2] += 1
            col[a > 0.6, :3] = C_HOT
            col[deg <= 1, :3] = C_ISO
        col[g["hot"], :3] = C_HOT
        col[:, 3] = np.clip(0.30 + 0.70 * a, 0, 1) * fade
        size = 2.5 + 9.0 * a
        if SHRINK_WITH_FADE:
            size = size * (0.30 + 0.70 * fade)
        size[g["hot"]] *= 2.2
        halo = col.copy(); halo[:, 3] *= 0.16
        self.it_halo.setData(pos=pos, color=halo, size=size * 4.5)
        self.it_core.setData(pos=pos, color=col, size=size)
        self.cur_labels = [(pos[i], self.label_text(g, i, self.j_cache),
                            col[i, :3], float(min(fade[i] * 1.4, 1.0)))
                           for i in self.pick_labels(g)]
        self.hud["nodes"].setText(
            f"{len(a)} / {len(g['e_part'])+len(g['e_harm'])+len(g['e_web'])}")

    # ------------------------------------------------------------------ teclas
    def keyPressEvent(self, ev):
        k, shift = ev.key(), ev.modifiers() & QtCore.Qt.ShiftModifier
        step = 10.0 if shift else 2.0
        keys = {QtCore.Qt.Key_1: 0, QtCore.Qt.Key_2: 1, QtCore.Qt.Key_3: 2,
                QtCore.Qt.Key_4: 3, QtCore.Qt.Key_5: 4}
        if k == QtCore.Qt.Key_Space:
            self.clock.toggle()
        elif k == QtCore.Qt.Key_Right:
            self.clock.seek(self.clock.t + step)
        elif k == QtCore.Qt.Key_Left:
            self.clock.seek(max(0.0, self.clock.t - step))
        elif k in keys:
            self.layout_mode = LAYOUTS[keys[k]]; self.j_cache = -10**9
        elif k == QtCore.Qt.Key_BracketRight:
            self.window_s = min(self.window_s * 1.25, 8.0); self.j_cache = -10**9
        elif k == QtCore.Qt.Key_BracketLeft:
            self.window_s = max(self.window_s / 1.25, 0.15); self.j_cache = -10**9
        elif k == QtCore.Qt.Key_R:
            self.autorotate = not self.autorotate
        elif k == QtCore.Qt.Key_V:
            self.split.setSizes([0, 1] if self.split.sizes()[0] else [420, 980])
        elif k == QtCore.Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self.timer.stop(); self.clock.close()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(ev)


# --------------------------------------------------------------------------- main
def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    path = VIDEO.strip() or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not path:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Elegi el video o audio del ave", "",
            "Audio/Video (*.mp4 *.mkv *.mov *.avi *.wav *.mp3 *.flac *.m4a);;Todos (*)")
    if not path:
        sys.exit("No elegiste archivo.")
    if not os.path.exists(path):
        sys.exit(f"No existe: {path}")

    print("[1/3] decodificando audio...", file=sys.stderr)
    x, sr = load_audio(path, SR)
    total = len(x) / sr
    i0 = int(max(START_AT, 0.0) * sr)
    i1 = len(x) if not DURATION else min(len(x), i0 + int(DURATION * sr))
    if i0 >= len(x):
        sys.exit(f"START_AT={START_AT}s pero el archivo dura {total:0.1f}s")
    if (i1 - i0) < len(x):
        print(f"      archivo de {total/60:0.1f} min -> analizo de "
              f"{START_AT:0.0f}s a {i1/sr:0.0f}s (edita START_AT / DURATION)",
              file=sys.stderr)
    x = np.ascontiguousarray(x[i0:i1])
    print("[2/3] STFT + picos...", file=sys.stderr)
    an = analyze(x, sr, N_FFT, HOP, FMIN, FMAX, FLOOR_DB, MAX_PEAKS, GATE_DB)
    print("[3/3] MFCC + PCA + chroma...", file=sys.stderr)
    feat = Features(an, sr)
    print(f"      {len(x)/sr:0.1f}s | {an.db.shape[1]} cuadros | "
          f"{sum(len(p[0]) for p in an.peaks)} picos | "
          f"PCA var {[round(float(v),3) for v in feat.explained]}", file=sys.stderr)

    win = LiveWindow(path, x, sr, an, feat, t_offset=i0 / sr)
    win.show()
    app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    main()
