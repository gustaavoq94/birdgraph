// jni_bridge.cpp -- puente entre Kotlin y el motor C++.
//
// Estrategia: el analisis vive del lado nativo y Kotlin solo guarda un handle.
// Los buffers del grafo se exponen como DirectByteBuffer, asi el renderer los
// sube a GL sin copiar ni generar basura para el GC 60 veces por segundo.
#include <jni.h>
#include <android/log.h>
#include "dsp.h"
#include "graph.h"
#include <memory>
#include <cstring>

#define LOG_TAG "BirdGraphNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

struct Session {
    bg::Analysis analysis;
    bg::GraphOut out;
    // buffers de tamano fijo para exponer a GL sin realocar
    std::vector<float> nodeBuf, webBuf, partBuf, harmBuf, labelPos, labelData;
};

inline Session* S(jlong h) { return reinterpret_cast<Session*>(h); }

struct ProgCtx { JNIEnv* env; jobject cb; jmethodID mid; };

void progressTrampoline(float p, void* user) {
    auto* c = static_cast<ProgCtx*>(user);
    if (c && c->cb && c->mid) c->env->CallVoidMethod(c->cb, c->mid, (jfloat)p);
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeAnalyze(
        JNIEnv* env, jobject, jfloatArray pcm, jint sr, jint nFft, jint hop,
        jfloat fmin, jfloat fmax, jfloat floorDb, jfloat gateDb, jint maxPeaks,
        jint nMels, jint nMfcc, jboolean useDeltas, jint smooth, jobject progressCb) {
    const jsize n = env->GetArrayLength(pcm);
    jfloat* data = env->GetFloatArrayElements(pcm, nullptr);

    bg::Params p;
    p.sr = sr; p.nFft = nFft; p.hop = hop;
    p.fmin = fmin; p.fmax = fmax;
    p.floorDb = floorDb; p.gateDb = gateDb; p.maxPeaks = maxPeaks;
    p.nMels = nMels; p.nMfcc = nMfcc; p.useDeltas = useDeltas; p.smooth = smooth;

    ProgCtx ctx{env, progressCb, nullptr};
    if (progressCb) {
        jclass cls = env->GetObjectClass(progressCb);
        ctx.mid = env->GetMethodID(cls, "onProgress", "(F)V");
    }

    auto* s = new Session();
    s->analysis = bg::analyze(data, (size_t)n, p,
                              progressCb ? progressTrampoline : nullptr, &ctx);
    env->ReleaseFloatArrayElements(pcm, data, JNI_ABORT);

    LOGI("analyze: %d frames, %d peaks, hop=%d",
         s->analysis.nFrames, (int)s->analysis.peakFreq.size(), s->analysis.p.hop);

    // reservar generosamente para no realocar durante el render
    s->nodeBuf.resize(4096 * 8);
    s->webBuf.resize(4096 * 11);
    s->partBuf.resize(4096 * 11);
    s->harmBuf.resize(4096 * 11);
    s->labelPos.resize(64 * 7);
    s->labelData.resize(64 * 8);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeFree(JNIEnv*, jobject, jlong h) {
    delete S(h);
}

JNIEXPORT jintArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetMeta(JNIEnv* env, jobject, jlong h) {
    auto& A = S(h)->analysis;
    jint v[4] = { A.nFrames, A.p.hop, A.p.sr, (jint)A.peakFreq.size() };
    jintArray r = env->NewIntArray(4);
    env->SetIntArrayRegion(r, 0, 4, v);
    return r;
}

// Todos los descriptores en un array plano: centroid, spread, flux, tonality,
// conc, rms (6 * nFrames) + explained (3) + refDb (1)
JNIEXPORT jfloatArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetDescriptors(JNIEnv* env, jobject, jlong h) {
    auto& A = S(h)->analysis;
    const int nf = A.nFrames;
    jfloatArray r = env->NewFloatArray(nf * 6 + 4);
    if (nf <= 0) return r;
    env->SetFloatArrayRegion(r, 0 * nf, nf, A.centroid.data());
    env->SetFloatArrayRegion(r, 1 * nf, nf, A.spread.data());
    env->SetFloatArrayRegion(r, 2 * nf, nf, A.flux.data());
    env->SetFloatArrayRegion(r, 3 * nf, nf, A.tonality.data());
    env->SetFloatArrayRegion(r, 4 * nf, nf, A.conc.data());
    env->SetFloatArrayRegion(r, 5 * nf, nf, A.rms.data());
    float tail[4] = { A.explained[0], A.explained[1], A.explained[2], A.refDb };
    env->SetFloatArrayRegion(r, 6 * nf, 4, tail);
    return r;
}

// Columna de espectro de un cuadro, ya normalizada 0..1 y submuestreada a `points`
JNIEXPORT jfloatArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetSpectrum(
        JNIEnv* env, jobject, jlong h, jint frame, jint points) {
    auto& A = S(h)->analysis;
    jfloatArray r = env->NewFloatArray(points);
    if (A.nFrames <= 0) return r;
    int j = frame < 0 ? 0 : (frame >= A.nFrames ? A.nFrames - 1 : frame);
    std::vector<float> v(points, 0.0f);
    const float lo = A.refDb - 60.0f;
    int b0 = 0, b1 = A.nBins - 1;
    while (b0 < A.nBins && A.freqs[b0] < A.p.fmin) ++b0;
    while (b1 > 0 && A.freqs[b1] > A.p.fmax) --b1;
    const int bw = b1 - b0 + 1;
    for (int i = 0; i < points; ++i) {
        int k0 = b0 + (int)((int64_t)i * bw / points);
        int k1 = b0 + (int)((int64_t)(i + 1) * bw / points);
        float m = -1e9f;
        for (int k = k0; k < std::max(k1, k0 + 1) && k <= b1; ++k)
            m = std::max(m, A.db[(size_t)j * A.nBins + k]);
        v[i] = std::min(1.0f, std::max(0.0f, (m - lo) / 60.0f));
    }
    env->SetFloatArrayRegion(r, 0, points, v.data());
    return r;
}

JNIEXPORT jfloatArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetChroma(
        JNIEnv* env, jobject, jlong h, jint frame) {
    auto& A = S(h)->analysis;
    jfloatArray r = env->NewFloatArray(12);
    if (A.nFrames <= 0) return r;
    int j = frame < 0 ? 0 : (frame >= A.nFrames ? A.nFrames - 1 : frame);
    env->SetFloatArrayRegion(r, 0, 12, &A.chroma[(size_t)j * 12]);
    return r;
}

JNIEXPORT jfloatArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetPca(
        JNIEnv* env, jobject, jlong h, jint frame) {
    auto& A = S(h)->analysis;
    jfloatArray r = env->NewFloatArray(3);
    if (A.nFrames <= 0) return r;
    int j = frame < 0 ? 0 : (frame >= A.nFrames ? A.nFrames - 1 : frame);
    env->SetFloatArrayRegion(r, 0, 3, &A.pca[(size_t)j * 3]);
    return r;
}

// Construye el grafo y copia a los buffers persistentes.
// Devuelve [nNodes, nWeb, nPartial, nHarm, nLabels, hotIndex].
JNIEXPORT jintArray JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeBuildGraph(
        JNIEnv* env, jobject, jlong h, jint frame, jint layout,
        jfloat windowSec, jfloat trailSec, jfloat fadeHalf, jfloat fadeHalfMan,
        jfloat alphaCutoff, jint nodeHop, jint knn, jfloat tolSemis,
        jfloat fluxPulse, jboolean colorChroma, jboolean shrinkFade,
        jint maxLabels, jfloat labelMinSep) {
    auto* s = S(h);
    bg::GraphParams gp;
    gp.layout = layout; gp.windowSec = windowSec; gp.trailSec = trailSec;
    gp.fadeHalf = fadeHalf; gp.fadeHalfMan = fadeHalfMan;
    gp.alphaCutoff = alphaCutoff; gp.nodeHop = nodeHop; gp.knn = knn;
    gp.tolSemis = tolSemis; gp.fluxPulse = fluxPulse;
    gp.colorChroma = colorChroma; gp.shrinkFade = shrinkFade;
    gp.maxLabels = maxLabels; gp.labelMinSep = labelMinSep;

    bg::buildGraph(s->analysis, frame, gp, s->out);

    auto copy = [](std::vector<float>& dst, const std::vector<float>& src) {
        if (dst.size() < src.size()) dst.resize(src.size());
        if (!src.empty()) std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
    };
    copy(s->nodeBuf,   s->out.nodes);
    copy(s->webBuf,    s->out.edgesWeb);
    copy(s->partBuf,   s->out.edgesPartial);
    copy(s->harmBuf,   s->out.edgesHarm);
    copy(s->labelPos,  s->out.labelPos);
    copy(s->labelData, s->out.labelData);

    jint v[6] = { s->out.nNodes, s->out.nWeb, s->out.nPartial,
                  s->out.nHarm, s->out.nLabels, s->out.hotIndex };
    jintArray r = env->NewIntArray(6);
    env->SetIntArrayRegion(r, 0, 6, v);
    return r;
}

// Buffers directos: se piden una sola vez y el renderer los reusa siempre.
// which: 0 nodes, 1 web, 2 partial, 3 harm, 4 labelPos, 5 labelData
JNIEXPORT jobject JNICALL
Java_com_birdgraph_app_NativeAnalyzer_nativeGetBuffer(
        JNIEnv* env, jobject, jlong h, jint which) {
    auto* s = S(h);
    std::vector<float>* v = nullptr;
    switch (which) {
        case 0: v = &s->nodeBuf;   break;
        case 1: v = &s->webBuf;    break;
        case 2: v = &s->partBuf;   break;
        case 3: v = &s->harmBuf;   break;
        case 4: v = &s->labelPos;  break;
        default: v = &s->labelData; break;
    }
    return env->NewDirectByteBuffer(v->data(), (jlong)(v->size() * sizeof(float)));
}

} // extern "C"
