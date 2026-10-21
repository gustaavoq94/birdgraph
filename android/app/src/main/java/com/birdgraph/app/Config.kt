package com.birdgraph.app

/**
 * Todos los parametros ajustables en un solo lugar, igual que el bloque CONFIG del
 * script de escritorio. Se pueden cambiar en caliente desde el panel de ajustes.
 */
object Config {
    // --- banda de analisis (ajustar por especie: koel 500-5000, jilguero 1500-9000)
    var fmin = 400f
    var fmax = 9000f

    // --- que tramo analizar. durationSec = 0 -> todo el archivo
    var startAtSec = 0f
    var durationSec = 180f

    // --- ventanas de tiempo
    var windowSec = 1.2f      // ventana visible en layouts de picos
    var trailSec = 6.0f       // estela en manifold

    // --- desvanecido: el brillo cae exponencialmente con la edad del nodo
    var fadeHalfLife = 0.45f          // vida media (s) en layouts de picos
    var fadeHalfLifeManifold = 1.6f   // mas largo: ahi interesa ver la estructura
    var alphaCutoff = 0.035f          // por debajo de esto el nodo ni se construye
    var shrinkWithFade = true

    // --- densidad / rendimiento
    var nodeHop = 3           // 1 = denso y lento, 4-6 = liviano
    var maxPeaks = 7
    var knn = 3
    var tolSemitones = 2.5f

    // --- STFT
    var sampleRate = 44100
    var nFft = 2048
    var hop = 256
    var floorDb = 14f
    var gateDb = 45f

    // --- MFCC / PCA
    var nMfcc = 40
    var nMels = 64
    var useDeltas = true
    var smooth = 5

    // --- estetica
    var layout = 0            // 0 manifold, 1 circle, 2 helix, 3 wall, 4 sphere
    var showVideo = true
    var showLabels = true
    var maxLabels = 9
    var labelMinSep = 0.16f
    var colorByChroma = true
    var fluxPulse = 0.18f
    var autoRotate = false

    val layoutNames = listOf("manifold", "circle", "helix", "wall", "sphere")
    val noteNames = listOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    fun hzToNote(hz: Float): String {
        if (hz <= 0f || !hz.isFinite()) return "--"
        val m = Math.round(69.0 + 12.0 * (Math.log((hz / 440.0)) / Math.log(2.0))).toInt()
        return "${noteNames[((m % 12) + 12) % 12]}${m / 12 - 1}"
    }
}
