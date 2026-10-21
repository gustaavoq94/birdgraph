// graph.h -- construccion del grafo por cuadro. Puerto de la clase Builder de
// birdgraph_live.py. Corre 60 veces por segundo, asi que reusa buffers y no aloca.
#pragma once
#include "dsp.h"
#include <vector>

namespace bg {

enum Layout { MANIFOLD = 0, CIRCLE = 1, HELIX = 2, WALL = 3, SPHERE = 4 };

struct GraphParams {
    int   layout      = MANIFOLD;
    float windowSec   = 1.2f;   // ventana visible en layouts de picos
    float trailSec    = 6.0f;   // estela en manifold
    float fadeHalf    = 0.45f;  // vida media del desvanecido (s)
    float fadeHalfMan = 1.6f;
    float alphaCutoff = 0.035f;
    int   nodeHop     = 3;
    int   knn         = 3;
    float tolSemis    = 2.5f;
    float fluxPulse   = 0.18f;
    bool  colorChroma = true;
    bool  shrinkFade  = true;
    int   maxLabels   = 9;
    float labelMinSep = 0.16f;
};

// Buffers listos para subir a GL. Se reusan entre cuadros.
struct GraphOut {
    // nodos: x,y,z,size,r,g,b,a
    std::vector<float> nodes;
    int nNodes = 0;
    // aristas instanciadas: x0,y0,z0,x1,y1,z1,width,r,g,b,a
    std::vector<float> edgesWeb, edgesPartial, edgesHarm;
    int nWeb = 0, nPartial = 0, nHarm = 0;
    // etiquetas: x,y,z,r,g,b,alpha + indice de cuadro + datos
    std::vector<float> labelPos;    // (nLabels * 7)
    std::vector<float> labelData;   // (nLabels * 8): freq,db,amp,ageMs,harm,frame,pcaOrPc,isHot
    int nLabels = 0;
    int hotIndex = 0;
};

void buildGraph(const Analysis& A, int frameIdx, const GraphParams& gp, GraphOut& out);

} // namespace bg
