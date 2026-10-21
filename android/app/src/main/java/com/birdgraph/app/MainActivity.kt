package com.birdgraph.app

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.collectAsState
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import com.birdgraph.app.gl.GraphRenderer
import com.birdgraph.app.ui.MainScreen
import com.birdgraph.app.ui.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private val vm: MainViewModel by viewModels()
    private lateinit var renderer: GraphRenderer

    private val picker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            // permiso persistente: si no, al rotar la pantalla se pierde el acceso
            runCatching {
                contentResolver.takePersistableUriPermission(
                    it, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            vm.load(it)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        renderer = GraphRenderer { vm.analyzerOrNull() }

        // "compartir con BirdGraph" desde la galeria
        if (intent?.action == android.content.Intent.ACTION_SEND) {
            val shared = if (android.os.Build.VERSION.SDK_INT >= 33)
                intent.getParcelableExtra(android.content.Intent.EXTRA_STREAM, Uri::class.java)
            else @Suppress("DEPRECATION")
                intent.getParcelableExtra<Uri>(android.content.Intent.EXTRA_STREAM)
            shared?.let { vm.load(it) }
        }

        setContent {
            val state by vm.state.collectAsState()
            val uri by vm.videoUri.collectAsState()
            MainScreen(
                state = state,
                analyzer = vm.analyzerOrNull(),
                renderer = renderer,
                player = vm.player,
                videoUri = uri,
                onPick = { picker.launch(arrayOf("video/*", "audio/*")) }
            )
        }
    }

    override fun onPause() {
        super.onPause()
        vm.player?.pause()
    }
}

class MainViewModel(app: android.app.Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow<UiState>(UiState.Idle)
    val state = _state.asStateFlow()

    private val _videoUri = MutableStateFlow<Uri?>(null)
    val videoUri = _videoUri.asStateFlow()

    private var analyzer: NativeAnalyzer? = null
    var player: ExoPlayer? = null
        private set

    fun analyzerOrNull(): NativeAnalyzer? = analyzer?.takeIf { it.isReady }

    fun load(uri: Uri) {
        viewModelScope.launch {
            _videoUri.value = uri
            _state.value = UiState.Working("Decodificando audio...", 0f)
            try {
                val ctx = getApplication<android.app.Application>()

                val pcm = withContext(Dispatchers.IO) {
                    AudioExtractor.extract(
                        ctx, uri, Config.startAtSec, Config.durationSec
                    ) { p ->
                        _state.value = UiState.Working("Decodificando audio...", p * 0.4f)
                    }.pcm
                }
                require(pcm.isNotEmpty()) { "No se pudo leer audio del archivo" }

                val a = withContext(Dispatchers.Default) {
                    NativeAnalyzer().also { an ->
                        an.analyze(pcm) { p ->
                            _state.value = UiState.Working(
                                "STFT, MFCC, PCA...", 0.4f + p * 0.6f)
                        }
                    }
                }
                analyzer?.close()
                analyzer = a

                // el reproductor es el reloj maestro de toda la sincronia
                player?.release()
                player = ExoPlayer.Builder(ctx).build().apply {
                    setMediaItem(MediaItem.fromUri(uri))
                    prepare()
                    if (Config.startAtSec > 0f) seekTo((Config.startAtSec * 1000).toLong())
                    repeatMode = ExoPlayer.REPEAT_MODE_ONE
                    playWhenReady = true
                }
                _state.value = UiState.Ready
            } catch (e: Exception) {
                _state.value = UiState.Failed(e.message ?: "Fallo al procesar el archivo")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        analyzer?.close()
        player?.release()
    }
}
