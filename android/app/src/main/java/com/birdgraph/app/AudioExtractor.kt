package com.birdgraph.app

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Decodifica la pista de audio de cualquier archivo (mp4, mkv, m4a, wav, mp3...)
 * a PCM mono float, remuestreado al sample rate de Config.
 *
 * Esto reemplaza a ffmpeg del script de escritorio: MediaExtractor + MediaCodec son
 * nativos de Android, asi que no hay binarios externos ni dependencias pesadas.
 */
object AudioExtractor {

    class Result(val pcm: FloatArray, val sampleRate: Int, val durationSec: Float)

    fun interface Progress { fun onProgress(p: Float) }

    fun extract(
        context: Context, uri: Uri,
        startSec: Float = 0f, durationSec: Float = 0f,
        progress: Progress? = null
    ): Result {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)

        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                track = i; format = f; break
            }
        }
        require(track >= 0 && format != null) { "El archivo no tiene pista de audio" }

        extractor.selectTrack(track)
        val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val totalUs = if (format.containsKey(MediaFormat.KEY_DURATION))
            format.getLong(MediaFormat.KEY_DURATION) else 0L

        val startUs = (startSec * 1_000_000L).toLong()
        val endUs = if (durationSec > 0f) startUs + (durationSec * 1_000_000L).toLong()
                    else Long.MAX_VALUE
        if (startUs > 0) extractor.seekTo(startUs, MediaExtractor.SEEK_TO_CLOSEST_SYNC)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val out = ArrayList<FloatArray>(256)
        var totalSamples = 0
        val info = MediaCodec.BufferInfo()
        var sawInputEnd = false
        var sawOutputEnd = false
        val spanUs = (if (endUs == Long.MAX_VALUE) (totalUs - startUs) else (endUs - startUs))
            .coerceAtLeast(1L)

        while (!sawOutputEnd) {
            if (!sawInputEnd) {
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val size = extractor.readSampleData(inBuf, 0)
                    if (size < 0 || extractor.sampleTime > endUs) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEnd = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outIdx = codec.dequeueOutputBuffer(info, 10_000)
            if (outIdx >= 0) {
                if (info.size > 0) {
                    val buf = codec.getOutputBuffer(outIdx)!!
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    val mono = toMonoFloat(buf, channels, codec.outputFormat)
                    if (mono.isNotEmpty()) { out.add(mono); totalSamples += mono.size }
                    if (progress != null && totalUs > 0) {
                        val done = (info.presentationTimeUs - startUs).toFloat() / spanUs
                        progress.onProgress(done.coerceIn(0f, 1f))
                    }
                }
                codec.releaseOutputBuffer(outIdx, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEnd = true
            }
        }

        codec.stop(); codec.release(); extractor.release()

        // aplanar
        val flat = FloatArray(totalSamples)
        var pos = 0
        for (chunk in out) { chunk.copyInto(flat, pos); pos += chunk.size }

        val target = Config.sampleRate
        val pcm = if (srcRate == target) flat else resample(flat, srcRate, target)
        return Result(pcm, target, pcm.size.toFloat() / target)
    }

    /** Mezcla a mono y normaliza a float en [-1,1], sea PCM 16 bit o float. */
    private fun toMonoFloat(buf: ByteBuffer, channels: Int, fmt: MediaFormat): FloatArray {
        val isFloat = fmt.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                fmt.getInteger(MediaFormat.KEY_PCM_ENCODING) == 4 // ENCODING_PCM_FLOAT
        buf.order(ByteOrder.nativeOrder())
        return if (isFloat) {
            val fb = buf.asFloatBuffer()
            val frames = fb.remaining() / channels
            FloatArray(frames) { i ->
                var s = 0f
                for (c in 0 until channels) s += fb.get(i * channels + c)
                s / channels
            }
        } else {
            val sb = buf.asShortBuffer()
            val frames = sb.remaining() / channels
            FloatArray(frames) { i ->
                var s = 0f
                for (c in 0 until channels) s += sb.get(i * channels + c) / 32768f
                s / channels
            }
        }
    }

    /** Remuestreo lineal. Suficiente: despues todo pasa por una STFT con ventana. */
    private fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (x.isEmpty()) return x
        val n = ((x.size.toLong() * to) / from).toInt().coerceAtLeast(1)
        val out = FloatArray(n)
        val step = (x.size - 1).toDouble() / (n - 1).coerceAtLeast(1)
        for (i in 0 until n) {
            val p = i * step
            val i0 = p.toInt()
            val i1 = min(i0 + 1, x.size - 1)
            val f = (p - i0).toFloat()
            out[i] = x[i0] * (1f - f) + x[i1] * f
        }
        return out
    }
}
