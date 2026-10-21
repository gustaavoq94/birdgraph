# BirdGraph Android

Puerto nativo del visualizador de cantos de aves. Elegís un video o audio de la
galería, la app lo analiza y muestra el grafo espectral 3D sincronizado con la
reproducción.

## Abrir el proyecto

1. Android Studio → **Open** → carpeta `BirdGraph`.
2. Android Studio va a avisar que falta el Gradle wrapper: aceptá cuando ofrezca
   generarlo (o corré `gradle wrapper` si tenés Gradle instalado). No incluyo el
   `.jar` binario del wrapper acá.
3. Instalá el **NDK** y **CMake** desde SDK Manager → SDK Tools si no los tenés.
4. Run. Requiere Android 8.0+ (API 26) por GLES 3.0 y PCM float en MediaCodec.

## Arquitectura

```
MediaExtractor + MediaCodec   →  PCM mono float, remuestreado    (AudioExtractor.kt)
        ↓
C++ nativo (una sola vez)     →  STFT, picos, MFCC, PCA, chroma  (dsp.cpp)
        ↓
C++ por cuadro (~0.2 ms)      →  grafo, layouts, fade, etiquetas (graph.cpp)
        ↓
DirectByteBuffer sin copia    →  GLES 3.0, blending aditivo      (GraphRenderer.kt)
        ↑
ExoPlayer.currentPosition     →  reloj maestro de la sincronía
```

**El DSP está verificado contra el Python.** Compilando `dsp.cpp` en escritorio y
comparándolo con `birdfeatures.py` sobre el mismo audio: correlación 1.000000 en
centroide, spread, flux, tonalidad y concentración cromática (error ~1e-4, que es
precisión de float32), **3146 picos detectados en ambos**, y varianza explicada por
PCA idéntica (22.3% / 18.0% / 12.3%).

## Decisiones técnicas que importan

**glLineWidth no sirve en GLES.** Casi todos los drivers lo limitan a 1.0. Las
aristas no se dibujan como líneas: cada una se expande a un quad orientado a la
cámara dentro del vertex shader, con instanced rendering. Eso da control real del
grosor y permite el degradado transversal que produce el glow.

**El glow es real**, no las capas simuladas de la versión matplotlib: blending
aditivo (`GL_ONE, GL_ONE`) con salida premultiplicada. Cada familia de aristas se
dibuja 3 veces con grosor decreciente y alpha creciente.

**Cero allocations por frame.** El grafo se construye en C++ sobre buffers
reusados y se expone a Kotlin como `DirectByteBuffer`. El renderer los sube a GL
sin copiar. Nada de basura para el GC 60 veces por segundo.

**El reloj maestro es ExoPlayer.** En escritorio hubo que construir uno desde el
stream de audio; acá `getCurrentPosition()` ya lo da, y de ahí sale el índice de
cuadro STFT. La sincronía sale más simple que en PC.

## Rendimiento

Reconstrucción del grafo medida en x86-64: **0.13–0.21 ms** según layout (en Python
eran 1.7 ms). En arm64 esperá algo similar o mejor con `-O3 -ffast-math`. Sobra
margen para 60 fps.

El análisis inicial sí tarda: unos segundos para 3 minutos de audio. Por eso corre
en `Dispatchers.Default` con barra de progreso, nunca en el hilo de UI.

## Config

`Config.kt` tiene todos los parámetros, igual que el bloque CONFIG del script de
escritorio. Los que más cambian el resultado:

| parámetro | default | efecto |
|---|---|---|
| `fmin` / `fmax` | 400 / 9000 | banda de análisis. Koel 500–5000, jilguero 1500–9000 |
| `durationSec` | 180 | cuánto analizar. 0 = todo (ojo con archivos largos) |
| `fadeHalfLife` | 0.45 | vida media del desvanecido en layouts de picos |
| `fadeHalfLifeManifold` | 1.6 | más largo: ahí interesa ver la estructura |
| `nodeHop` | 3 | 1 = denso y lento, 4–6 = liviano |
| `maxPeaks` | 7 | nodos por cuadro |

Desvanecido, estela, ventana, pulso, etiquetas y color se ajustan **en vivo** desde
el botón AJUSTES. La banda de frecuencia se aplica al abrir el archivo, porque
cambia el análisis completo.

## Los cinco layouts

Se cambian con la botonera arriba a la derecha.

1. **manifold** — nodos = cuadros proyectados por PCA de sus MFCC. La línea de
   trayectoria es temporal; las cuerdas verdes unen puntos cercanos en timbre pero
   lejanos en el tiempo: ahí se ven los motivos repetidos del canto.
2. **circle** — tiempo → radio, frecuencia → ángulo. El spread abre y cierra el anillo.
3. **helix** — ángulo = clase de altura, radio = octava, profundidad = tiempo.
4. **wall** — x = tiempo, y = log-frecuencia, z = amplitud. El más legible.
5. **sphere** — azimut = altura, elevación = octava, radio = antigüedad.

## Gestos

Un dedo orbita. Dos dedos hacen zoom. La botonera de abajo tiene play/pausa, ±5s,
auto-rotación, mostrar/ocultar video, ajustes y abrir otro archivo.

## Limitaciones conocidas

- El picker pide `video/*` y `audio/*`; formatos exóticos que MediaCodec no soporte
  van a fallar con mensaje de error (no crash).
- El remuestreo es lineal. Suficiente porque después todo pasa por una STFT con
  ventana, pero no es un resampler de calidad de audio.
- Sin micrófono en vivo. La arquitectura lo permitiría (`AudioRecord` + análisis
  incremental), pero el PCA necesita ajustarse sobre una ventana inicial primero.
