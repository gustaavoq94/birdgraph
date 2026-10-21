// dsp.cpp -- implementacion. Ver dsp.h.
#include "dsp.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace bg {

static const float PI = 3.14159265358979323846f;

// ---------------------------------------------------------------- FFT radix-2
// Cooley-Tukey iterativo in-place. nFft es potencia de 2 (2048 por defecto).
class FFT {
public:
    explicit FFT(int n) : n_(n) {
        rev_.resize(n);
        int bits = 0; while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            rev_[i] = r;
        }
        cos_.resize(n / 2); sin_.resize(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            cos_[i] = std::cos(-2.0f * PI * i / n);
            sin_[i] = std::sin(-2.0f * PI * i / n);
        }
    }
    // re/im de largo n. Transformada compleja in-place.
    void run(float* re, float* im) const {
        for (int i = 0; i < n_; ++i) {
            int j = rev_[i];
            if (j > i) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= n_; len <<= 1) {
            int half = len >> 1, step = n_ / len;
            for (int i = 0; i < n_; i += len) {
                for (int k = 0; k < half; ++k) {
                    float wr = cos_[k * step], wi = sin_[k * step];
                    int a = i + k, b = a + half;
                    float tr = re[b] * wr - im[b] * wi;
                    float ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr;        im[a] += ti;
                }
            }
        }
    }
    int size() const { return n_; }
private:
    int n_;
    std::vector<int> rev_;
    std::vector<float> cos_, sin_;
};

// ---------------------------------------------------------------- mel / DCT
static inline float hzToMel(float f) { return 2595.0f * std::log10(1.0f + f / 700.0f); }
static inline float melToHz(float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); }

// Banco triangular con normalizacion Slaney (misma area por filtro).
static std::vector<float> melFilterbank(int sr, int nFft, int nMels,
                                        float fmin, float fmax, int nBins) {
    std::vector<float> fb((size_t)nMels * nBins, 0.0f);
    std::vector<float> edges(nMels + 2);
    float m0 = hzToMel(fmin), m1 = hzToMel(fmax);
    for (int i = 0; i < nMels + 2; ++i)
        edges[i] = melToHz(m0 + (m1 - m0) * i / (nMels + 1));
    for (int m = 0; m < nMels; ++m) {
        float lo = edges[m], ctr = edges[m + 1], hi = edges[m + 2];
        float enorm = 2.0f / std::max(hi - lo, 1e-9f);
        for (int k = 0; k < nBins; ++k) {
            float f = (float)k * sr / nFft;
            float l = (f - lo) / std::max(ctr - lo, 1e-9f);
            float r = (hi - f) / std::max(hi - ctr, 1e-9f);
            float v = std::min(l, r);
            fb[(size_t)m * nBins + k] = (v > 0.0f ? v : 0.0f) * enorm;
        }
    }
    return fb;
}

// DCT-II ortonormal (igual a scipy.fft.dct(type=2, norm="ortho")).
static void dctII(const float* in, float* out, int n, int nOut,
                  const std::vector<float>& table) {
    for (int k = 0; k < nOut; ++k) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += in[i] * table[(size_t)k * n + i];
        float scale = (k == 0) ? std::sqrt(1.0f / (4.0f * n)) : std::sqrt(1.0f / (2.0f * n));
        out[k] = 2.0f * s * scale;
    }
}

// ---------------------------------------------------------------- PCA
// Eigendescomposicion de la covarianza por Jacobi ciclico. D = 78, es barato.
// Los autovectores de la covarianza son los vectores singulares derechos del SVD,
// asi que da lo mismo que np.linalg.svd sobre la matriz centrada.
static void jacobiEigen(std::vector<double>& A, int n,
                        std::vector<double>& V, int sweeps = 60) {
    V.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(size_t)i * n + i] = 1.0;
    for (int s = 0; s < sweeps; ++s) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += A[(size_t)i * n + j] * A[(size_t)i * n + j];
        if (off < 1e-12) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = A[(size_t)p * n + q];
                if (std::fabs(apq) < 1e-14) continue;
                double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                for (int k = 0; k < n; ++k) {
                    double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - sn * akq;
                    A[(size_t)k * n + q] = sn * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - sn * aqk;
                    A[(size_t)q * n + k] = sn * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double vkp = V[(size_t)k * n + p], vkq = V[(size_t)k * n + q];
                    V[(size_t)k * n + p] = c * vkp - sn * vkq;
                    V[(size_t)k * n + q] = sn * vkp + c * vkq;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- helpers
static float percentile(std::vector<float> v, float q) {
    if (v.empty()) return 0.0f;
    size_t k = (size_t)(q / 100.0f * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int fitMemory(size_t nSamples, int nFft, int hop, float limitMb) {
    int bins = nFft / 2 + 1;
    while ((double)nSamples / hop * bins * 4.0 / 1e6 > limitMb && hop < nFft) hop *= 2;
    return hop;
}

// ---------------------------------------------------------------- analyze
Analysis analyze(const float* x, size_t n, const Params& pin,
                 ProgressFn progress, void* user) {
    Analysis A;
    A.p = pin;
    A.p.hop = fitMemory(n, pin.nFft, pin.hop);
    const Params& p = A.p;

    const int nFft = p.nFft, hop = p.hop, nBins = nFft / 2 + 1;
    A.nBins = nBins;

    // ventana hann periodica (fftbins=True en scipy)
    std::vector<float> win(nFft);
    for (int i = 0; i < nFft; ++i)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / nFft));

    // padding reflect, igual que en el Python
    const int pad = nFft / 2;
    std::vector<float> xp((size_t)n + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        xp[i] = (n > (size_t)pad) ? x[pad - i] : 0.0f;
        xp[n + pad + i] = (n > (size_t)pad) ? x[n - 2 - i] : 0.0f;
    }
    std::memcpy(xp.data() + pad, x, n * sizeof(float));

    int nFrames = (int)((xp.size() >= (size_t)nFft) ? 1 + (xp.size() - nFft) / hop : 0);
    if (nFrames <= 0) return A;
    A.nFrames = nFrames;

    A.freqs.resize(nBins);
    for (int k = 0; k < nBins; ++k) A.freqs[k] = (float)k * p.sr / nFft;

    // ---- STFT (magnitud). Se guarda mag y db.
    std::vector<float> mag((size_t)nBins * nFrames);
    A.db.resize((size_t)nBins * nFrames);
    FFT fft(nFft);
    std::vector<float> re(nFft), im(nFft);
    for (int j = 0; j < nFrames; ++j) {
        const float* src = xp.data() + (size_t)j * hop;
        for (int i = 0; i < nFft; ++i) { re[i] = src[i] * win[i]; im[i] = 0.0f; }
        fft.run(re.data(), im.data());
        float* mcol = &mag[(size_t)j * nBins];
        float* dcol = &A.db[(size_t)j * nBins];
        for (int k = 0; k < nBins; ++k) {
            float m = std::sqrt(re[k] * re[k] + im[k] * im[k]);
            mcol[k] = m;
            dcol[k] = 20.0f * std::log10(m + 1e-9f);
        }
        if (progress && (j % 256) == 0) progress(0.35f * j / nFrames, user);
    }

    // ---- descriptores por cuadro
    A.centroid.resize(nFrames); A.spread.resize(nFrames);
    A.flux.resize(nFrames);     A.tonality.resize(nFrames);
    A.rms.resize(nFrames);
    for (int j = 0; j < nFrames; ++j) {
        const float* m = &mag[(size_t)j * nBins];
        double sum = 0.0, cen = 0.0, logSum = 0.0, ari = 0.0;
        for (int k = 0; k < nBins; ++k) {
            sum += m[k]; cen += (double)A.freqs[k] * m[k];
            logSum += std::log((double)m[k] + 1e-9); ari += m[k];
        }
        cen /= (sum + 1e-12);
        double var = 0.0;
        for (int k = 0; k < nBins; ++k) {
            double d = A.freqs[k] - cen;
            var += d * d * m[k] / (sum + 1e-12);
        }
        A.centroid[j] = (float)cen;
        A.spread[j]   = (float)std::sqrt(std::max(var, 0.0));
        // flatness = media geometrica / media aritmetica, en dB
        double geo = std::exp(logSum / nBins), am = ari / nBins + 1e-9;
        A.tonality[j] = (float)(10.0 * std::log10(geo / am));
        if (j == 0) A.flux[0] = 0.0f;
        else {
            const float* pm = &mag[(size_t)(j - 1) * nBins];
            double s = 0.0;
            for (int k = 0; k < nBins; ++k) { double d = m[k] - pm[k]; if (d > 0) s += d * d; }
            A.flux[j] = (float)std::sqrt(s);
        }
        // rms del bloque de audio correspondiente
        size_t s0 = (size_t)j * hop;
        double e = 0.0; int cnt = 0;
        for (int i = 0; i < hop && s0 + i < n; ++i) { e += (double)x[s0 + i] * x[s0 + i]; ++cnt; }
        A.rms[j] = cnt ? (float)std::sqrt(e / cnt) : 0.0f;
    }

    // ---- picos espectrales (los nodos del grafo)
    int b0 = 0, b1 = nBins - 1;
    while (b0 < nBins && A.freqs[b0] < p.fmin) ++b0;
    while (b1 > 0 && A.freqs[b1] > p.fmax) --b1;
    const int bw = std::max(b1 - b0 + 1, 1);

    A.refDb = -1e9f;
    for (int j = 0; j < nFrames; ++j)
        for (int k = b0; k <= b1; ++k) A.refDb = std::max(A.refDb, A.db[(size_t)j * nBins + k]);

    A.active.assign(nFrames, 0);
    A.peakOffset.assign(nFrames + 1, 0);
    std::vector<float> band(bw);
    std::vector<int> cand; std::vector<float> prom;
    for (int j = 0; j < nFrames; ++j) {
        const float* dcol = &A.db[(size_t)j * nBins];
        float colMax = -1e9f;
        for (int k = 0; k < bw; ++k) { band[k] = dcol[b0 + k]; colMax = std::max(colMax, band[k]); }
        A.peakOffset[j + 1] = A.peakOffset[j];
        if (colMax < A.refDb - p.gateDb) continue;    // silencio: sin nodos
        A.active[j] = 1;
        float floorDb = std::max(percentile(band, 60.0f) + p.floorDb, A.refDb - p.gateDb);

        cand.clear(); prom.clear();
        for (int k = 1; k < bw - 1; ++k) {
            if (band[k] <= floorDb) continue;
            if (!(band[k] > band[k - 1] && band[k] >= band[k + 1])) continue;
            // prominencia estilo scipy: bajar a cada lado hasta encontrar algo mas alto
            float lmin = band[k], rmin = band[k];
            for (int i = k - 1; i >= 0; --i) { if (band[i] > band[k]) break; lmin = std::min(lmin, band[i]); }
            for (int i = k + 1; i < bw; ++i) { if (band[i] > band[k]) break; rmin = std::min(rmin, band[i]); }
            float pr = band[k] - std::max(lmin, rmin);
            if (pr < 6.0f) continue;
            cand.push_back(k); prom.push_back(band[k]);
        }
        // separacion minima de 2 bins: los mas fuertes ganan
        std::vector<int> order(cand.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return prom[a] > prom[b]; });
        std::vector<int> keep;
        for (int oi : order) {
            if ((int)keep.size() >= p.maxPeaks) break;
            bool ok = true;
            for (int kk : keep) if (std::abs(cand[oi] - cand[kk]) < 2) { ok = false; break; }
            if (ok) keep.push_back(oi);
        }
        std::sort(keep.begin(), keep.end(),
                  [&](int a, int b) { return cand[a] < cand[b]; });
        for (int oi : keep) {
            int k = cand[oi];
            // interpolacion parabolica -> frecuencia sub-bin
            float a = band[k - 1], b = band[k], c = band[k + 1];
            float dn = a - 2 * b + c;
            float delta = (std::fabs(dn) > 1e-9f) ? 0.5f * (a - c) / dn : 0.0f;
            delta = std::max(-0.5f, std::min(0.5f, delta));
            A.peakFreq.push_back(((float)(b0 + k) + delta) * p.sr / nFft);
            A.peakDb.push_back(b);
            A.peakOffset[j + 1]++;
        }
    }
    if (progress) progress(0.55f, user);

    // ---- chroma (12 clases de altura) + concentracion
    A.chroma.assign((size_t)12 * nFrames, 0.0f);
    A.conc.resize(nFrames);
    A.pitchClass.resize(nFrames);
    std::vector<int> pc(nBins, -1);
    for (int k = b0; k <= b1; ++k) {
        if (A.freqs[k] <= 0) continue;
        int m = (int)std::lround(12.0 * std::log2(A.freqs[k] / 440.0) + 69.0);
        pc[k] = ((m % 12) + 12) % 12;
    }
    for (int j = 0; j < nFrames; ++j) {
        float* ch = &A.chroma[(size_t)j * 12];
        const float* m = &mag[(size_t)j * nBins];
        for (int k = b0; k <= b1; ++k) if (pc[k] >= 0) ch[pc[k]] += m[k] * m[k];
        float s = 0.0f; for (int c = 0; c < 12; ++c) s += ch[c];
        s += 1e-12f;
        double h = 0.0; int best = 0; float bv = -1.0f;
        for (int c = 0; c < 12; ++c) {
            ch[c] /= s;
            double q = std::max((double)ch[c], 1e-12);
            h -= q * std::log(q);
            if (ch[c] > bv) { bv = ch[c]; best = c; }
        }
        A.conc[j] = (float)std::max(0.0, std::min(1.0, 1.0 - h / std::log(12.0)));
        A.pitchClass[j] = (uint8_t)best;
    }

    // ---- MFCC (+ deltas), descartando c0 (energia global, no timbre)
    const int nMels = p.nMels, nMfcc = p.nMfcc;
    std::vector<float> fb = melFilterbank(p.sr, nFft, nMels, p.fmin, p.fmax, nBins);
    std::vector<float> dctTable((size_t)nMfcc * nMels);
    for (int k = 0; k < nMfcc; ++k)
        for (int i = 0; i < nMels; ++i)
            dctTable[(size_t)k * nMels + i] = std::cos(PI * k * (2 * i + 1) / (2.0f * nMels));

    const int D = (nMfcc - 1) * (p.useDeltas ? 2 : 1);   // sin c0
    std::vector<float> coef((size_t)nFrames * nMfcc);
    std::vector<float> logmel(nMels), tmp(nMfcc);
    for (int j = 0; j < nFrames; ++j) {
        const float* m = &mag[(size_t)j * nBins];
        for (int b = 0; b < nMels; ++b) {
            const float* row = &fb[(size_t)b * nBins];
            double s = 0.0;
            for (int k = 0; k < nBins; ++k) s += (double)row[k] * m[k] * m[k];
            logmel[b] = (float)std::log(s + 1e-10);
        }
        dctII(logmel.data(), tmp.data(), nMels, nMfcc, dctTable);
        std::memcpy(&coef[(size_t)j * nMfcc], tmp.data(), nMfcc * sizeof(float));
        if (progress && (j % 256) == 0) progress(0.55f + 0.25f * j / nFrames, user);
    }

    std::vector<float> X((size_t)nFrames * D);
    for (int j = 0; j < nFrames; ++j) {
        float* row = &X[(size_t)j * D];
        for (int k = 1; k < nMfcc; ++k) row[k - 1] = coef[(size_t)j * nMfcc + k];
        if (p.useDeltas) {   // np.gradient sobre el eje temporal
            int jm = std::max(j - 1, 0), jp = std::min(j + 1, nFrames - 1);
            float denom = (float)(jp - jm); if (denom <= 0) denom = 1.0f;
            for (int k = 1; k < nMfcc; ++k)
                row[nMfcc - 1 + k - 1] =
                    (coef[(size_t)jp * nMfcc + k] - coef[(size_t)jm * nMfcc + k]) / denom;
        }
    }

    // ---- PCA a 3D, ajustado SOLO con cuadros activos (el silencio no define los ejes)
    std::vector<int> act;
    for (int j = 0; j < nFrames; ++j) if (A.active[j]) act.push_back(j);
    if (act.empty()) { act.resize(nFrames); std::iota(act.begin(), act.end(), 0); }

    std::vector<double> mu(D, 0.0);
    for (int j : act) for (int k = 0; k < D; ++k) mu[k] += X[(size_t)j * D + k];
    for (int k = 0; k < D; ++k) mu[k] /= act.size();

    std::vector<double> cov((size_t)D * D, 0.0);
    std::vector<double> z(D);
    for (int j : act) {
        for (int k = 0; k < D; ++k) z[k] = X[(size_t)j * D + k] - mu[k];
        for (int a = 0; a < D; ++a) {
            double za = z[a];
            if (za == 0.0) continue;
            for (int b = a; b < D; ++b) cov[(size_t)a * D + b] += za * z[b];
        }
    }
    double nrm = std::max((double)act.size() - 1.0, 1.0);
    for (int a = 0; a < D; ++a)
        for (int b = a; b < D; ++b) {
            cov[(size_t)a * D + b] /= nrm;
            cov[(size_t)b * D + a] = cov[(size_t)a * D + b];
        }

    std::vector<double> evec;
    std::vector<double> covCopy = cov;
    jacobiEigen(covCopy, D, evec);
    std::vector<std::pair<double,int>> ev(D);
    for (int i = 0; i < D; ++i) ev[i] = { covCopy[(size_t)i * D + i], i };
    std::sort(ev.begin(), ev.end(), [](auto& a, auto& b){ return a.first > b.first; });
    double totVar = 0.0; for (auto& e : ev) totVar += std::max(e.first, 0.0);

    A.explained.resize(3);
    A.pca.assign((size_t)nFrames * 3, 0.0f);
    double sc[3];
    for (int c = 0; c < 3; ++c) {
        double lam = std::max(ev[c].first, 0.0);
        A.explained[c] = (float)(lam / (totVar + 1e-12));
        sc[c] = std::max(std::sqrt(lam), 1e-6);   // whitening, igual que en Python
    }
    // el signo de un autovector es arbitrario; lo fijo para que el resultado sea
    // reproducible entre corridas y plataformas
    for (int c = 0; c < 3; ++c) {
        int col = ev[c].second;
        int big = 0; double bv = 0.0;
        for (int k = 0; k < D; ++k) {
            double v = std::fabs(evec[(size_t)k * D + col]);
            if (v > bv) { bv = v; big = k; }
        }
        if (evec[(size_t)big * D + col] < 0.0)
            for (int k = 0; k < D; ++k) evec[(size_t)k * D + col] = -evec[(size_t)k * D + col];
    }
    for (int j = 0; j < nFrames; ++j) {
        for (int c = 0; c < 3; ++c) {
            int col = ev[c].second;
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += (X[(size_t)j * D + k] - mu[k]) * evec[(size_t)k * D + col];
            A.pca[(size_t)j * 3 + c] = (float)(s / sc[c]);
        }
        if (progress && (j % 512) == 0) progress(0.80f + 0.15f * j / nFrames, user);
    }

    // suavizado de la trayectoria (media movil) -> curvas continuas
    if (p.smooth > 1 && nFrames > p.smooth) {
        std::vector<float> sm = A.pca;
        int k = p.smooth, h = k / 2;
        for (int j = 0; j < nFrames; ++j)
            for (int c = 0; c < 3; ++c) {
                double s = 0.0;
                for (int t = -h; t <= h; ++t) {
                    int jj = j + t;
                    if (jj >= 0 && jj < nFrames) s += A.pca[(size_t)jj * 3 + c];
                }
                sm[(size_t)j * 3 + c] = (float)(s / k);
            }
        A.pca.swap(sm);
    }

    // normalizar al percentil 97 de los cuadros activos -> rango util ~[-1,1]
    std::vector<float> absv;
    absv.reserve(act.size() * 3);
    for (int j : act) for (int c = 0; c < 3; ++c) absv.push_back(std::fabs(A.pca[(size_t)j * 3 + c]));
    float r = std::max(percentile(absv, 97.0f), 1e-6f);
    for (auto& v : A.pca) v /= r;

    if (progress) progress(1.0f, user);
    return A;
}

} // namespace bg
