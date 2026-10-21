package com.birdgraph.app.gl

/**
 * En GLES glLineWidth esta limitado a 1.0 en casi todos los drivers, asi que las
 * aristas no se dibujan como lineas: cada una se expande a un quad orientado a la
 * camara dentro del vertex shader (instanced rendering). Eso da control real del
 * grosor y ademas permite el degradado transversal que produce el glow.
 */
object Shaders {

    const val EDGE_VS = """#version 300 es
layout(location = 0) in vec2 aCorner;   // x: 0..1 a lo largo, y: -1..1 a lo ancho
layout(location = 1) in vec3 aP0;       // instancia: extremo inicial
layout(location = 2) in vec3 aP1;       // instancia: extremo final
layout(location = 3) in float aWidth;
layout(location = 4) in vec4 aColor;

uniform mat4  uMVP;
uniform vec2  uViewport;
uniform float uWidthScale;
uniform float uAlphaScale;

out vec4  vColor;
out float vAcross;

void main() {
    vec4 c0 = uMVP * vec4(aP0, 1.0);
    vec4 c1 = uMVP * vec4(aP1, 1.0);

    // direccion del segmento en pixeles de pantalla
    vec2 s0 = (c0.xy / max(abs(c0.w), 1e-5)) * uViewport * 0.5;
    vec2 s1 = (c1.xy / max(abs(c1.w), 1e-5)) * uViewport * 0.5;
    vec2 dir = s1 - s0;
    float len = length(dir);
    dir = (len > 1e-5) ? dir / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);

    vec4 base = mix(c0, c1, aCorner.x);
    float w = max(aWidth * uWidthScale, 0.6);
    vec2 offsetPx = nrm * (aCorner.y * w * 0.5);
    base.xy += offsetPx / (uViewport * 0.5) * abs(base.w);

    gl_Position = base;
    vColor = vec4(aColor.rgb, aColor.a * uAlphaScale);
    vAcross = aCorner.y;
}
"""

    // salida premultiplicada + blending aditivo = glow real, no capas simuladas
    const val EDGE_FS = """#version 300 es
precision mediump float;
in vec4  vColor;
in float vAcross;
out vec4 fragColor;

void main() {
    float a = 1.0 - abs(vAcross);
    a = a * a;                       // caida suave hacia los bordes del quad
    fragColor = vec4(vColor.rgb * vColor.a * a, 1.0);
}
"""

    const val NODE_VS = """#version 300 es
layout(location = 0) in vec3  aPos;
layout(location = 1) in float aSize;
layout(location = 2) in vec4  aColor;

uniform mat4  uMVP;
uniform float uSizeScale;
uniform float uAlphaScale;
uniform float uDensity;

out vec4 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = max(aSize * uSizeScale * uDensity, 1.0);
    vColor = vec4(aColor.rgb, aColor.a * uAlphaScale);
}
"""

    const val NODE_FS = """#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 fragColor;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d) * 2.0;
    if (r > 1.0) discard;
    float a = exp(-3.2 * r * r);     // perfil gaussiano: nucleo brillante, halo suave
    fragColor = vec4(vColor.rgb * vColor.a * a, 1.0);
}
"""

    const val GRID_VS = """#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

    const val GRID_FS = """#version 300 es
precision mediump float;
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor.rgb * uColor.a, 1.0); }
"""
}
