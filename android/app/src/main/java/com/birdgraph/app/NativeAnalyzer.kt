package com.birdgraph.app

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Envoltorio del motor C++. El analisis vive del lado nativo; aca solo se guarda
 * un handle. Los buffers del grafo son DirectByteBuffer compartidos, asi el
 * renderer los sube a GL sin copiar ni generar basura para el GC.
 */
class NativeAnalyzer : AutoCloseable {

    companion object {
        init { System.loadLibrary("birdgraph") }
        const val LAYOUT_MANIFOLD = 0
        const val LAYOUT_CIRCLE = 1
        const val LAYOUT_HELIX = 2
        const val LAYOUT_WALL = 3
        const val LAYOUT_SPHERE = 4
    }

    fun interface Progress { fun onProgress(p: Float) }

    private var handle: Long = 0L

    var frameCount = 0; private set
    var hop = 0; private set
    var sampleRate = 0; private set
    var peakCount = 0; private set
    var refDb = 0f; private set
    var explained = FloatArray(3); private set

    private lateinit var centroid: FloatArray
    private lateinit var spread: FloatArray
    private lateinit var flux: FloatArray
    private lateinit var tonality: FloatArray
    private lateinit var conc: FloatArray
    private lateinit var rms: FloatArray
    private var fluxMax = 1f

    // buffers GL persistentes
    lateinit var nodeBuf: FloatBuffer; private set
    lateinit var webBuf: FloatBuffer; private set
    lateinit var partBuf: FloatBuffer; private set
    lateinit var harmBuf: FloatBuffer; private set
    lateinit var labelPosBuf: FloatBuffer; private set
    lateinit var labelDataBuf: FloatBuffer; private set

    // contadores del ultimo buildGraph
    var nNodes = 0; private set
    var nWeb = 0; private set
    var nPartial = 0; private set
    var nHarm = 0; private set
    var nLabels = 0; private set
    var hotIndex = 0; private set

    val isReady get() = handle != 0L

    fun analyze(pcm: FloatArray, progress: Progress? = null) {
        close()
        handle = nativeAnalyze(
            pcm, Config.sampleRate, Config.nFft, Config.hop,
            Config.fmin, Config.fmax, Config.floorDb, Config.gateDb, Config.maxPeaks,
            Config.nMels, Config.nMfcc, Config.useDeltas, Config.smooth, progress
        )
        require(handle != 0L) { "El analisis nativo fallo" }

        val meta = nativeGetMeta(handle)
        frameCount = meta[0]; hop = meta[1]; sampleRate = meta[2]; peakCount = meta[3]

        val d = nativeGetDescriptors(handle)
        val n = frameCount
        centroid = d.copyOfRange(0, n)
        spread = d.copyOfRange(n, 2 * n)
        flux = d.copyOfRange(2 * n, 3 * n)
        tonality = d.copyOfRange(3 * n, 4 * n)
        conc = d.copyOfRange(4 * n, 5 * n)
        rms = d.copyOfRange(5 * n, 6 * n)
        explained = d.copyOfRange(6 * n, 6 * n + 3)
        refDb = d[6 * n + 3]
        fluxMax = (flux.maxOrNull() ?: 1f).coerceAtLeast(1e-9f)

        nodeBuf = buf(0); webBuf = buf(1); partBuf = buf(2)
        harmBuf = buf(3); labelPosBuf = buf(4); labelDataBuf = buf(5)
    }

    private fun buf(which: Int): FloatBuffer =
        (nativeGetBuffer(handle, which) as ByteBuffer)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()

    /** Reconstruye el grafo para el cuadro dado. ~0.2 ms. */
    fun buildGraph(frame: Int, layout: Int) {
        if (handle == 0L) return
        val r = nativeBuildGraph(
            handle, frame, layout,
            Config.windowSec, Config.trailSec,
            Config.fadeHalfLife, Config.fadeHalfLifeManifold, Config.alphaCutoff,
            Config.nodeHop, Config.knn, Config.tolSemitones, Config.fluxPulse,
            Config.colorByChroma, Config.shrinkWithFade,
            Config.maxLabels, Config.labelMinSep
        )
        nNodes = r[0]; nWeb = r[1]; nPartial = r[2]
        nHarm = r[3]; nLabels = r[4]; hotIndex = r[5]
    }

    fun frameAt(seconds: Float): Int =
        if (hop <= 0) 0
        else ((seconds * sampleRate / hop).toInt()).coerceIn(0, (frameCount - 1).coerceAtLeast(0))

    fun centroidAt(j: Int) = centroid.getOrElse(j) { 0f }
    fun spreadAt(j: Int) = spread.getOrElse(j) { 0f }
    fun fluxAt(j: Int) = flux.getOrElse(j) { 0f } / fluxMax
    fun tonalityAt(j: Int) = tonality.getOrElse(j) { 0f }
    fun concAt(j: Int) = conc.getOrElse(j) { 0f }
    fun rmsAt(j: Int) = rms.getOrElse(j) { 0f }

    fun spectrum(frame: Int, points: Int): FloatArray =
        if (handle == 0L) FloatArray(points) else nativeGetSpectrum(handle, frame, points)

    fun chroma(frame: Int): FloatArray =
        if (handle == 0L) FloatArray(12) else nativeGetChroma(handle, frame)

    fun pca(frame: Int): FloatArray =
        if (handle == 0L) FloatArray(3) else nativeGetPca(handle, frame)

    override fun close() {
        if (handle != 0L) { nativeFree(handle); handle = 0L }
    }

    // ------------------------------------------------------------------ JNI
    private external fun nativeAnalyze(
        pcm: FloatArray, sr: Int, nFft: Int, hop: Int,
        fmin: Float, fmax: Float, floorDb: Float, gateDb: Float, maxPeaks: Int,
        nMels: Int, nMfcc: Int, useDeltas: Boolean, smooth: Int, progress: Progress?
    ): Long

    private external fun nativeFree(handle: Long)
    private external fun nativeGetMeta(handle: Long): IntArray
    private external fun nativeGetDescriptors(handle: Long): FloatArray
    private external fun nativeGetSpectrum(handle: Long, frame: Int, points: Int): FloatArray
    private external fun nativeGetChroma(handle: Long, frame: Int): FloatArray
    private external fun nativeGetPca(handle: Long, frame: Int): FloatArray
    private external fun nativeGetBuffer(handle: Long, which: Int): Any
    private external fun nativeBuildGraph(
        handle: Long, frame: Int, layout: Int,
        windowSec: Float, trailSec: Float, fadeHalf: Float, fadeHalfMan: Float,
        alphaCutoff: Float, nodeHop: Int, knn: Int, tolSemis: Float,
        fluxPulse: Float, colorChroma: Boolean, shrinkFade: Boolean,
        maxLabels: Int, labelMinSep: Float
    ): IntArray
}
