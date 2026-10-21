package com.birdgraph.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdgraph.app.Config
import com.birdgraph.app.NativeAnalyzer
import com.birdgraph.app.gl.GraphRenderer
import kotlin.math.max

private val Mono = FontFamily.Monospace
private val Dim = Color(0xFF9FB3C8)
private val Bright = Color(0xFFE9F2FB)

/**
 * Etiquetas de datos flotando sobre los nodos. Se proyectan de 3D a pantalla con la
 * MVP del renderer en cada recomposicion, asi siguen al nodo mientras se orbita.
 */
@Composable
fun LabelOverlay(
    renderer: GraphRenderer,
    analyzer: NativeAnalyzer?,
    tick: Int,
    modifier: Modifier = Modifier
) {
    if (!Config.showLabels || analyzer == null || !analyzer.isReady) return
    @Suppress("UNUSED_EXPRESSION") tick   // fuerza recomposicion cada frame de UI

    val density = LocalDensity.current
    val n = analyzer.nLabels
    if (n <= 0) return

    val pos = analyzer.labelPosBuf
    val data = analyzer.labelDataBuf
    val scr = FloatArray(2)

    Box(modifier) {
        for (i in 0 until n) {
            val px = pos.get(i * 7); val py = pos.get(i * 7 + 1); val pz = pos.get(i * 7 + 2)
            val r = pos.get(i * 7 + 3); val g = pos.get(i * 7 + 4); val b = pos.get(i * 7 + 5)
            val alpha = pos.get(i * 7 + 6)
            if (alpha < 0.12f) continue
            if (!renderer.project(px, py, pz, scr)) continue
            if (scr[0] < -80 || scr[0] > renderer.viewportW + 80) continue
            if (scr[1] < -80 || scr[1] > renderer.viewportH + 80) continue

            val freq = data.get(i * 8); val db = data.get(i * 8 + 1)
            val amp = data.get(i * 8 + 2); val ageMs = data.get(i * 8 + 3)
            val harm = data.get(i * 8 + 4).toInt(); val frame = data.get(i * 8 + 5).toInt()
            val conc = data.get(i * 8 + 6); val isHot = data.get(i * 8 + 7) > 0.5f

            val text = buildString {
                if (Config.layout == NativeAnalyzer.LAYOUT_MANIFOLD) {
                    val p = analyzer.pca(frame)
                    val t = frame.toFloat() * analyzer.hop / analyzer.sampleRate
                    append("t %6.2fs\n".format(t))
                    append("PC %+.2f %+.2f %+.2f\n".format(p[0], p[1], p[2]))
                    append("chr %-2s conc %.2f".format(Config.noteNames[harm % 12], conc))
                    if (isHot) {
                        append("\ncen %.2fk  spr %.2fk".format(
                            analyzer.centroidAt(frame) / 1000f, analyzer.spreadAt(frame) / 1000f))
                        append("\nflx %.3f  ton %.1f".format(
                            analyzer.fluxAt(frame), analyzer.tonalityAt(frame)))
                    }
                } else {
                    append("%6.3f kHz  %s\n".format(freq / 1000f, Config.hzToNote(freq)))
                    append("%6.1f dB   a %.2f\n".format(db, amp))
                    append("age %4.0f ms".format(ageMs))
                    if (harm >= 1 && !isHot) append("\narmonico h$harm")
                    if (isHot) {
                        append("\ncen %.2fk  spr %.2fk".format(
                            analyzer.centroidAt(frame) / 1000f, analyzer.spreadAt(frame) / 1000f))
                        append("\nflx %.3f  ton %.1f".format(
                            analyzer.fluxAt(frame), analyzer.tonalityAt(frame)))
                    }
                }
            }

            val xDp = with(density) { (scr[0] + 12f).toDp() }
            val yDp = with(density) { (scr[1] - 18f).toDp() }
            Box(
                Modifier
                    .offset(x = xDp, y = yDp)
                    .background(Color(0f, 0f, 0f, 0.55f * alpha + 0.2f))
                    .padding(horizontal = 4.dp, vertical = 2.dp)
            ) {
                Text(
                    text = text,
                    color = Color(r, g, b, alpha.coerceIn(0f, 1f)),
                    fontSize = 8.sp, fontFamily = Mono, lineHeight = 10.sp
                )
            }
        }
    }
}

/** Espectro en vivo con escala logaritmica de frecuencia. */
@Composable
fun SpectrumStrip(values: FloatArray, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        if (values.isEmpty()) return@Canvas
        val w = size.width; val h = size.height
        val step = w / values.size
        for (i in values.indices) {
            val v = values[i].coerceIn(0f, 1f)
            drawRect(
                color = Color(0.14f, 1f, 0.60f, 0.25f + 0.75f * v),
                topLeft = Offset(i * step, h * (1f - v)),
                size = Size(max(step, 1f), h * v)
            )
        }
    }
}

/** Barras de chroma: energia por clase de altura. */
@Composable
fun ChromaBars(values: FloatArray, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        if (values.isEmpty()) return@Canvas
        val mx = (values.maxOrNull() ?: 1f).coerceAtLeast(1e-9f)
        val w = size.width / 12f
        for (i in 0 until 12) {
            val v = (values[i] / mx).coerceIn(0f, 1f)
            drawRect(
                color = Color(0.91f, 0.95f, 0.98f, 0.25f + 0.6f * v),
                topLeft = Offset(i * w + w * 0.15f, size.height * (1f - v)),
                size = Size(w * 0.7f, size.height * v)
            )
        }
    }
}

/** Una celda del HUD: etiqueta chica arriba, valor grande abajo. */
@Composable
fun HudCell(label: String, value: String) {
    Column(Modifier.padding(end = 14.dp)) {
        Text(label, color = Dim, fontSize = 8.sp, fontFamily = Mono)
        Text(value, color = Bright, fontSize = 12.sp, fontFamily = Mono)
    }
}
