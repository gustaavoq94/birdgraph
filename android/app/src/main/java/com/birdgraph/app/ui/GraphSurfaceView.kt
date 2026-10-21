package com.birdgraph.app.ui

import android.annotation.SuppressLint
import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent
import com.birdgraph.app.gl.GraphRenderer
import kotlin.math.hypot

/**
 * Superficie GL con gestos: un dedo orbita, dos dedos hacen zoom (pinch).
 * Se dibuja de forma continua para que el desvanecido se vea fluido aunque el
 * grafo solo se reconstruya cada nodeHop cuadros.
 */
@SuppressLint("ViewConstructor")
class GraphSurfaceView(
    context: Context,
    val renderer: GraphRenderer
) : GLSurfaceView(context) {

    private var lastX = 0f
    private var lastY = 0f
    private var lastDist = 0f
    private var mode = 0   // 0 nada, 1 orbita, 2 zoom

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
        renderer.setDensity(resources.displayMetrics.density.coerceIn(1f, 3f))
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                mode = 1; lastX = e.x; lastY = e.y
                parent?.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (e.pointerCount >= 2) { mode = 2; lastDist = dist(e) }
            }
            MotionEvent.ACTION_MOVE -> when {
                mode == 2 && e.pointerCount >= 2 -> {
                    val d = dist(e)
                    if (lastDist > 1f && d > 1f) {
                        renderer.distance = (renderer.distance * lastDist / d)
                            .coerceIn(1.2f, 16f)
                    }
                    lastDist = d
                }
                mode == 1 -> {
                    renderer.azimuth -= (e.x - lastX) * 0.32f
                    renderer.elevation = (renderer.elevation + (e.y - lastY) * 0.28f)
                        .coerceIn(-88f, 88f)
                    lastX = e.x; lastY = e.y
                }
            }
            MotionEvent.ACTION_POINTER_UP -> {
                mode = if (e.pointerCount <= 2) 0 else 2
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                mode = 0
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        }
        return true
    }

    private fun dist(e: MotionEvent) =
        hypot(e.getX(0) - e.getX(1), e.getY(0) - e.getY(1))
}
