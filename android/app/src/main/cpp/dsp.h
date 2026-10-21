// dsp.h -- analisis espectral de cantos de aves.
// Puerto 1:1 de birdfeatures.py / birdgraph.py. Sin dependencias externas.
#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace bg {

struct Params {
    int   sr        = 44100;
    int   nFft      = 2048;
    int   hop       = 256;
    float fmin      = 400.0f;
    float fmax      = 9000.0f;
    float floorDb   = 14.0f;   // umbral sobre el piso de ruido para aceptar un pico
    float gateDb    = 45.0f;   // cuanto por debajo del pico global es silencio
    int   maxPeaks  = 7;
    int   nMels     = 64;
    int   nMfcc     = 40;
    bool  useDeltas = true;
    int   smooth    = 5;       // suavizado de la trayectoria PCA (cuadros)
};

// Resultado del analisis completo. Todo indexado por cuadro (frame).
struct Analysis {
    Params p;
    int nFrames = 0;
    int nBins   = 0;
    float refDb = 0.0f;          // pico global (referencia de amplitud)

    std::vector<float> freqs;    // (nBins)
    std::vector<float> db;       // (nBins * nFrames) column-major: db[bin + frame*nBins]

    // descriptores por cuadro
    std::vector<float> centroid, spread, flux, tonality, rms;

    // picos espectrales: CSR-like para evitar vector<vector>
    std::vector<int>   peakOffset;  // (nFrames+1)
    std::vector<float> peakFreq;    // frecuencia interpolada sub-bin
    std::vector<float> peakDb;

    // features de alto nivel
    std::vector<float> pca;         // (nFrames * 3) normalizado ~[-1,1]
    std::vector<float> explained;   // (3) varianza explicada
    std::vector<float> chroma;      // (12 * nFrames) column-major
    std::vector<float> conc;        // (nFrames) concentracion cromatica
    std::vector<uint8_t> pitchClass;// (nFrames) clase de altura dominante
    std::vector<uint8_t> active;    // (nFrames) 1 si supera el gate
};

// x: PCM mono float. progress: 0..1, puede ser null.
using ProgressFn = void (*)(float, void*);
Analysis analyze(const float* x, size_t n, const Params& p,
                 ProgressFn progress = nullptr, void* user = nullptr);

// Sube el hop si el analisis no entraria en memoria (archivos largos).
int fitMemory(size_t nSamples, int nFft, int hop, float limitMb = 320.0f);

} // namespace bg
