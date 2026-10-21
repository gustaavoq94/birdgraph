package com.birdgraph.app.ui

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.birdgraph.app.Config
import com.birdgraph.app.NativeAnalyzer
import com.birdgraph.app.gl.GraphRenderer
import kotlinx.coroutines.delay

private val Mono = FontFamily.Monospace
private val Dim = Color(0xFF9FB3C8)
private val Accent = Color(0xFF25FF9A)

sealed class UiState {
    object Idle : UiState()
    data class Working(val step: String, val progress: Float) : UiState()
    object Ready : UiState()
    data class Failed(val message: String) : UiState()
}

@Composable
fun MainScreen(
    state: UiState,
    analyzer: NativeAnalyzer?,
    renderer: GraphRenderer,
    player: ExoPlayer?,
    videoUri: Uri?,
    onPick: () -> Unit
) {
    Column(
        Modifier
            .fillMaxSize()
            .background(Color(0xFF08080A))
            .systemBarsPadding()
    ) {
        when (state) {
            is UiState.Idle -> EmptyState(onPick, null)
            is UiState.Failed -> EmptyState(onPick, state.message)
            is UiState.Working -> WorkingState(state)
            is UiState.Ready -> ReadyState(analyzer, renderer, player, videoUri, onPick)
        }
    }
}

@Composable
private fun EmptyState(onPick: () -> Unit, error: String?) {
    Column(
        Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("birdgraph", color = Accent, fontSize = 30.sp, fontFamily = Mono)
        Spacer(Modifier.height(8.dp))
        Text(
            "Grafo espectral 3D del canto de aves.\n" +
            "Elegi un video o audio de la galeria.",
            color = Dim, fontSize = 13.sp, fontFamily = Mono
        )
        if (error != null) {
            Spacer(Modifier.height(14.dp))
            Text(error, color = Color(0xFFFF5577), fontSize = 12.sp, fontFamily = Mono)
        }
        Spacer(Modifier.height(26.dp))
        Button(onClick = onPick,
            colors = ButtonDefaults.buttonColors(containerColor = Accent,
                contentColor = Color.Black)) {
            Text("Elegir archivo", fontFamily = Mono)
        }
    }
}

@Composable
private fun WorkingState(state: UiState.Working) {
    Column(
        Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(state.step, color = Dim, fontSize = 13.sp, fontFamily = Mono)
        Spacer(Modifier.height(16.dp))
        LinearProgressIndicator(
            progress = { state.progress.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(0.8f),
            color = Accent, trackColor = Color(0xFF1D2530)
        )
        Spacer(Modifier.height(8.dp))
        Text("${(state.progress * 100).toInt()}%", color = Dim,
            fontSize = 11.sp, fontFamily = Mono)
    }
}

@Composable
private fun ColumnScope.ReadyState(
    analyzer: NativeAnalyzer?,
    renderer: GraphRenderer,
    player: ExoPlayer?,
    videoUri: Uri?,
    onPick: () -> Unit
) {
    if (analyzer == null) return
    var tick by remember { mutableIntStateOf(0) }
    var positionSec by remember { mutableFloatStateOf(0f) }
    var isPlaying by remember { mutableStateOf(true) }
    var showSettings by remember { mutableStateOf(false) }
    // Config.layout es un var plano: Compose no lo observa, asi que se espeja en
    // un state local para que los botones de layout recompongan la barra
    var layoutSel by remember { mutableIntStateOf(Config.layout) }
    val durationSec = remember(analyzer) {
        analyzer.frameCount.toFloat() * analyzer.hop / analyzer.sampleRate
    }

    // el reloj maestro es el reproductor: de ahi sale el cuadro de analisis
    LaunchedEffect(player) {
        while (true) {
            player?.let {
                positionSec = it.currentPosition / 1000f - Config.startAtSec
                isPlaying = it.isPlaying
            }
            renderer.currentFrame = analyzer.frameAt(positionSec.coerceAtLeast(0f))
            renderer.layout = Config.layout
            tick++
            delay(16)
        }
    }

    val j = renderer.currentFrame
    val spectrum = remember(j) { analyzer.spectrum(j, 128) }
    val chroma = remember(j) { analyzer.chroma(j) }

    Box(Modifier.weight(1f).fillMaxWidth()) {
        AndroidView(
            factory = { GraphSurfaceView(it, renderer) },
            modifier = Modifier.fillMaxSize(),
            onRelease = { it.onPause() },
            onReset = null
        )
        LabelOverlay(renderer, analyzer, tick, Modifier.fillMaxSize())

        if (Config.showVideo && videoUri != null && player != null) {
            AndroidView(
                factory = { ctx ->
                    PlayerView(ctx).apply {
                        useController = false
                        this.player = player
                    }
                },
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(10.dp)
                    .fillMaxWidth(0.34f)
                    .aspectRatio(16f / 9f)
                    .background(Color.Black)
            )
        }

        // botonera de layouts
        Row(
            Modifier.align(Alignment.TopEnd).padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Config.layoutNames.forEachIndexed { i, name ->
                val sel = layoutSel == i
                Text(
                    name.take(3).uppercase(),
                    color = if (sel) Color.Black else Dim,
                    fontSize = 9.sp, fontFamily = Mono,
                    modifier = Modifier
                        .background(if (sel) Accent else Color(0x33FFFFFF))
                        .clickable { layoutSel = i; Config.layout = i }
                        .padding(horizontal = 6.dp, vertical = 3.dp)
                )
            }
        }
    }

    // tiras de espectro y chroma
    Row(Modifier.fillMaxWidth().height(52.dp).padding(horizontal = 10.dp)) {
        SpectrumStrip(spectrum, Modifier.weight(1f).fillMaxHeight())
        Spacer(Modifier.width(8.dp))
        ChromaBars(chroma, Modifier.width(96.dp).fillMaxHeight())
    }

    // scrubber
    Slider(
        value = positionSec.coerceIn(0f, durationSec),
        onValueChange = { player?.seekTo(((it + Config.startAtSec) * 1000).toLong()) },
        valueRange = 0f..durationSec.coerceAtLeast(0.1f),
        colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Accent),
        modifier = Modifier.padding(horizontal = 10.dp)
    )

    // HUD desplazable
    Row(
        Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        val pca = analyzer.pca(j)
        HudCell("TIME", "%6.2f s".format(positionSec + Config.startAtSec))
        HudCell("CENTROID", "%s %.2fk".format(
            Config.hzToNote(analyzer.centroidAt(j)), analyzer.centroidAt(j) / 1000f))
        HudCell("SPREAD kHz", "%.2f".format(analyzer.spreadAt(j) / 1000f))
        HudCell("FLUX", "%.3f".format(analyzer.fluxAt(j)))
        HudCell("TONALITY dB", "%.1f".format(analyzer.tonalityAt(j)))
        HudCell("CHROMA CONC", "%.3f".format(analyzer.concAt(j)))
        HudCell("PCA xyz", "%+.2f %+.2f %+.2f".format(pca[0], pca[1], pca[2]))
        HudCell("VAR EXPL", analyzer.explained.joinToString(" ") {
            "${(it * 100).toInt()}%" })
        HudCell("NODES/EDGES", "${analyzer.nNodes}/${
            analyzer.nWeb + analyzer.nPartial + analyzer.nHarm}")
        HudCell("PEAKS", "${analyzer.peakCount}")
    }

    // controles
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Ctl(if (isPlaying) "PAUSA" else "PLAY") {
            player?.let { if (it.isPlaying) it.pause() else it.play() }
        }
        Ctl("-5s") { player?.seekTo(((positionSec - 5 + Config.startAtSec) * 1000)
            .toLong().coerceAtLeast(0)) }
        Ctl("+5s") { player?.seekTo(((positionSec + 5 + Config.startAtSec) * 1000).toLong()) }
        Ctl(if (Config.autoRotate) "ROT ON" else "ROT") {
            Config.autoRotate = !Config.autoRotate }
        Ctl(if (Config.showVideo) "VIDEO" else "VID OFF") {
            Config.showVideo = !Config.showVideo }
        Ctl("AJUSTES") { showSettings = true }
        Ctl("ABRIR") { onPick() }
    }

    if (showSettings) SettingsSheet(onDismiss = { showSettings = false })
}

@Composable
private fun Ctl(label: String, onClick: () -> Unit) {
    Text(
        label, color = Dim, fontSize = 10.sp, fontFamily = Mono,
        modifier = Modifier
            .background(Color(0x22FFFFFF))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 5.dp)
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsSheet(onDismiss: () -> Unit) {
    var fade by remember { mutableFloatStateOf(Config.fadeHalfLife) }
    var trail by remember { mutableFloatStateOf(Config.trailSec) }
    var window by remember { mutableFloatStateOf(Config.windowSec) }
    var labels by remember { mutableStateOf(Config.showLabels) }
    var chroma by remember { mutableStateOf(Config.colorByChroma) }
    var pulse by remember { mutableFloatStateOf(Config.fluxPulse) }

    ModalBottomSheet(onDismissRequest = onDismiss, containerColor = Color(0xFF101318)) {
        Column(Modifier.padding(18.dp).padding(bottom = 24.dp)) {
            Text("Ajustes en vivo", color = Accent, fontSize = 15.sp, fontFamily = Mono)
            Spacer(Modifier.height(12.dp))

            SettingSlider("Desvanecido (vida media)", fade, 0.1f, 3f, "%.2f s") {
                fade = it; Config.fadeHalfLife = it
            }
            SettingSlider("Estela manifold", trail, 1f, 15f, "%.1f s") {
                trail = it; Config.trailSec = it
            }
            SettingSlider("Ventana (layouts de picos)", window, 0.2f, 5f, "%.2f s") {
                window = it; Config.windowSec = it
            }
            SettingSlider("Pulso con flux", pulse, 0f, 0.6f, "%.2f") {
                pulse = it; Config.fluxPulse = it
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(labels, {
                    labels = it; Config.showLabels = it
                }, colors = SwitchDefaults.colors(checkedTrackColor = Accent))
                Text("  Etiquetas de datos", color = Dim, fontSize = 12.sp, fontFamily = Mono)
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(chroma, {
                    chroma = it; Config.colorByChroma = it
                }, colors = SwitchDefaults.colors(checkedTrackColor = Accent))
                Text("  Color por clase de altura", color = Dim,
                    fontSize = 12.sp, fontFamily = Mono)
            }
            Spacer(Modifier.height(10.dp))
            Text(
                "La banda de analisis (${Config.fmin.toInt()}-${Config.fmax.toInt()} Hz) " +
                "se aplica al abrir el archivo. Para el koel probá 500-5000.",
                color = Dim, fontSize = 10.sp, fontFamily = Mono
            )
        }
    }
}

@Composable
private fun SettingSlider(
    label: String, value: Float, min: Float, max: Float, fmt: String,
    onChange: (Float) -> Unit
) {
    Column(Modifier.padding(bottom = 10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, color = Dim, fontSize = 11.sp, fontFamily = Mono)
            Text(fmt.format(value), color = Color.White, fontSize = 11.sp, fontFamily = Mono)
        }
        Slider(
            value = value, onValueChange = onChange, valueRange = min..max,
            colors = SliderDefaults.colors(thumbColor = Color.White,
                activeTrackColor = Accent)
        )
    }
}
