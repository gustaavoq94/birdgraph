package com.birdgraph.app.gl

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.birdgraph.app.Config
import com.birdgraph.app.NativeAnalyzer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.sin

/**
 * Renderer del grafo. Blending aditivo sobre fondo negro: el glow es real, no las
 * capas superpuestas que usaba la version de matplotlib.
 *
 * Cada familia de aristas se dibuja 3 veces con grosor decreciente y alpha creciente
 * (halo ancho tenue -> nucleo fino brillante), que es lo que da el aspecto neon.
 */
class GraphRenderer(private val analyzer: () -> NativeAnalyzer?) : GLSurfaceView.Renderer {

    // ---- camara orbital
    @Volatile var azimuth = -58f
    @Volatile var elevation = 16f
    @Volatile var distance = 4.4f
    @Volatile var currentFrame = 0
    @Volatile var layout = 0

    private val proj = FloatArray(16)
    private val view = FloatArray(16)
    private val mvp = FloatArray(16)
    /** Copia de la MVP para que la UI pueda proyectar las etiquetas. */
    val mvpSnapshot = FloatArray(16)
    @Volatile var viewportW = 1
    @Volatile var viewportH = 1

    private var edgeProg = 0
    private var nodeProg = 0
    private var gridProg = 0

    private var quadVbo = 0
    private var edgeVbo = 0
    private var nodeVbo = 0
    private var gridVbo = 0
    private var edgeVao = 0
    private var nodeVao = 0
    private var gridVao = 0
    private var gridVerts = 0

    private var density = 1f
    private var lastBuilt = -1_000_000

    fun setDensity(d: Float) { density = d }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.031f, 0.031f, 0.039f, 1f)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE)   // aditivo

        edgeProg = link(Shaders.EDGE_VS, Shaders.EDGE_FS)
        nodeProg = link(Shaders.NODE_VS, Shaders.NODE_FS)
        gridProg = link(Shaders.GRID_VS, Shaders.GRID_FS)

        // quad base para las aristas instanciadas (triangle strip)
        val quad = floatArrayOf(0f, -1f, 0f, 1f, 1f, -1f, 1f, 1f)
        quadVbo = makeBuffer(quad)
        edgeVbo = genBuffer()
        nodeVbo = genBuffer()

        val ids = IntArray(3)
        GLES30.glGenVertexArrays(3, ids, 0)
        edgeVao = ids[0]; nodeVao = ids[1]; gridVao = ids[2]

        setupEdgeVao()
        setupNodeVao()
        setupGrid()
    }

    private fun setupEdgeVao() {
        GLES30.glBindVertexArray(edgeVao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 8, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, edgeVbo)
        val stride = 11 * 4
        // p0, p1, width, color -> todos por instancia
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 3, GLES30.GL_FLOAT, false, stride, 0)
        GLES30.glVertexAttribDivisor(1, 1)
        GLES30.glEnableVertexAttribArray(2)
        GLES30.glVertexAttribPointer(2, 3, GLES30.GL_FLOAT, false, stride, 12)
        GLES30.glVertexAttribDivisor(2, 1)
        GLES30.glEnableVertexAttribArray(3)
        GLES30.glVertexAttribPointer(3, 1, GLES30.GL_FLOAT, false, stride, 24)
        GLES30.glVertexAttribDivisor(3, 1)
        GLES30.glEnableVertexAttribArray(4)
        GLES30.glVertexAttribPointer(4, 4, GLES30.GL_FLOAT, false, stride, 28)
        GLES30.glVertexAttribDivisor(4, 1)
        GLES30.glBindVertexArray(0)
    }

    private fun setupNodeVao() {
        GLES30.glBindVertexArray(nodeVao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, nodeVbo)
        val stride = 8 * 4
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, stride, 0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 1, GLES30.GL_FLOAT, false, stride, 12)
        GLES30.glEnableVertexAttribArray(2)
        GLES30.glVertexAttribPointer(2, 4, GLES30.GL_FLOAT, false, stride, 16)
        GLES30.glBindVertexArray(0)
    }

    private fun setupGrid() {
        val lines = ArrayList<Float>()
        val n = 8; val half = 2f; val z = -1.4f
        for (i in 0..n) {
            val t = -half + 2 * half * i / n
            lines.addAll(listOf(t, -half, z, t, half, z))
            lines.addAll(listOf(-half, t, z, half, t, z))
        }
        gridVerts = lines.size / 3
        gridVbo = makeBuffer(lines.toFloatArray())
        GLES30.glBindVertexArray(gridVao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, gridVbo)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 12, 0)
        GLES30.glBindVertexArray(0)
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        viewportW = w; viewportH = h
        GLES30.glViewport(0, 0, w, h)
        Matrix.perspectiveM(proj, 0, 45f, w.toFloat() / h.coerceAtLeast(1), 0.1f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        val a = analyzer() ?: return
        if (!a.isReady) return

        if (Config.autoRotate) azimuth += 0.12f

        // camara orbital
        val ar = Math.toRadians(azimuth.toDouble())
        val er = Math.toRadians(elevation.toDouble())
        val ex = (distance * cos(er) * cos(ar)).toFloat()
        val ey = (distance * cos(er) * sin(ar)).toFloat()
        val ez = (distance * sin(er)).toFloat()
        Matrix.setLookAtM(view, 0, ex, ey, ez, 0f, 0f, 0f, 0f, 0f, 1f)
        Matrix.multiplyMM(mvp, 0, proj, 0, view, 0)
        System.arraycopy(mvp, 0, mvpSnapshot, 0, 16)

        // reconstruir el grafo solo cuando cambio el cuadro de analisis
        if (kotlin.math.abs(currentFrame - lastBuilt) >= Config.nodeHop) {
            lastBuilt = currentFrame
            a.buildGraph(currentFrame, layout)
        }

        drawGrid()

        // orden: hilos finos, luego parciales, luego armonicos arriba
        drawEdges(a.webBuf, a.nWeb, floatArrayOf(1f), floatArrayOf(1f))
        drawEdges(a.partBuf, a.nPartial,
            floatArrayOf(7f, 3f, 1.4f), floatArrayOf(0.055f, 0.13f, 0.95f))
        drawEdges(a.harmBuf, a.nHarm,
            floatArrayOf(9f, 4f, 1.7f), floatArrayOf(0.07f, 0.16f, 0.95f))
        drawNodes(a.nodeBuf, a.nNodes)
    }

    private fun drawGrid() {
        GLES30.glUseProgram(gridProg)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(gridProg, "uMVP"), 1, false, mvp, 0)
        GLES30.glUniform4f(GLES30.glGetUniformLocation(gridProg, "uColor"),
            0.23f, 0.27f, 0.33f, 0.10f)
        GLES30.glBindVertexArray(gridVao)
        GLES30.glDrawArrays(GLES30.GL_LINES, 0, gridVerts)
        GLES30.glBindVertexArray(0)
    }

    private fun drawEdges(buf: FloatBuffer, count: Int, widths: FloatArray, alphas: FloatArray) {
        if (count <= 0) return
        buf.position(0); buf.limit(count * 11)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, edgeVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, count * 11 * 4, buf, GLES30.GL_DYNAMIC_DRAW)
        buf.limit(buf.capacity())

        GLES30.glUseProgram(edgeProg)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(edgeProg, "uMVP"), 1, false, mvp, 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(edgeProg, "uViewport"),
            viewportW.toFloat(), viewportH.toFloat())
        val locW = GLES30.glGetUniformLocation(edgeProg, "uWidthScale")
        val locA = GLES30.glGetUniformLocation(edgeProg, "uAlphaScale")

        GLES30.glBindVertexArray(edgeVao)
        for (i in widths.indices) {
            GLES30.glUniform1f(locW, widths[i] * density)
            GLES30.glUniform1f(locA, alphas[i])
            GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLE_STRIP, 0, 4, count)
        }
        GLES30.glBindVertexArray(0)
    }

    private fun drawNodes(buf: FloatBuffer, count: Int) {
        if (count <= 0) return
        buf.position(0); buf.limit(count * 8)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, nodeVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, count * 8 * 4, buf, GLES30.GL_DYNAMIC_DRAW)
        buf.limit(buf.capacity())

        GLES30.glUseProgram(nodeProg)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(nodeProg, "uMVP"), 1, false, mvp, 0)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(nodeProg, "uDensity"), density)
        val locS = GLES30.glGetUniformLocation(nodeProg, "uSizeScale")
        val locA = GLES30.glGetUniformLocation(nodeProg, "uAlphaScale")

        GLES30.glBindVertexArray(nodeVao)
        GLES30.glUniform1f(locS, 4.5f); GLES30.glUniform1f(locA, 0.16f)  // halo
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, count)
        GLES30.glUniform1f(locS, 1.0f); GLES30.glUniform1f(locA, 1.0f)   // nucleo
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, count)
        GLES30.glBindVertexArray(0)
    }

    // ------------------------------------------------------------------ utils
    private fun genBuffer(): Int {
        val b = IntArray(1); GLES30.glGenBuffers(1, b, 0); return b[0]
    }

    private fun makeBuffer(data: FloatArray): Int {
        val id = genBuffer()
        val fb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().put(data).apply { position(0) }
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, id)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.size * 4, fb, GLES30.GL_STATIC_DRAW)
        return id
    }

    private fun link(vs: String, fs: String): Int {
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vs))
        GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fs))
        GLES30.glLinkProgram(p)
        val st = IntArray(1)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, st, 0)
        check(st[0] != 0) { "Link fallo: ${GLES30.glGetProgramInfoLog(p)}" }
        return p
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val st = IntArray(1)
        GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, st, 0)
        check(st[0] != 0) { "Compilacion fallo: ${GLES30.glGetShaderInfoLog(s)}" }
        return s
    }

    /** Proyecta un punto 3D a coordenadas de pantalla. Devuelve null si queda detras. */
    fun project(x: Float, y: Float, z: Float, out: FloatArray): Boolean {
        val m = mvpSnapshot
        val cw = m[3] * x + m[7] * y + m[11] * z + m[15]
        if (cw <= 1e-5f) return false
        val cx = m[0] * x + m[4] * y + m[8] * z + m[12]
        val cy = m[1] * x + m[5] * y + m[9] * z + m[13]
        out[0] = (cx / cw * 0.5f + 0.5f) * viewportW
        out[1] = (0.5f - cy / cw * 0.5f) * viewportH
        return true
    }
}
