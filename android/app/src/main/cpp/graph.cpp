// graph.cpp -- ver graph.h
#include "graph.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace bg {

static const float PI = 3.14159265358979323846f;

// paleta (RGB 0..1), igual que en la version de escritorio
static const float C_WEB[3]     = {0.72f, 0.80f, 0.90f};
static const float C_PARTIAL[3] = {1.00f, 0.18f, 0.48f};
static const float C_HARM[3]    = {0.14f, 1.00f, 0.60f};
static const float C_HOT[3]     = {1.00f, 1.00f, 1.00f};
static const float C_ISO[3]     = {0.95f, 1.00f, 0.27f};

// clase de altura -> RGB por circulo de quintas (notas vecinas armonicamente
// reciben colores vecinos, que es lo que hace legible el manifold)
static void chromaRgb(int pc, float* rgb) {
    float h = ((pc * 7) % 12) / 12.0f;
    float s = 0.78f, v = 1.0f;
    int i = (int)(h * 6.0f) % 6;
    float f = h * 6.0f - std::floor(h * 6.0f);
    float p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    switch (i) {
        case 0: rgb[0]=v; rgb[1]=t; rgb[2]=p; break;
        case 1: rgb[0]=q; rgb[1]=v; rgb[2]=p; break;
        case 2: rgb[0]=p; rgb[1]=v; rgb[2]=t; break;
        case 3: rgb[0]=p; rgb[1]=q; rgb[2]=v; break;
        case 4: rgb[0]=t; rgb[1]=p; rgb[2]=v; break;
        default:rgb[0]=v; rgb[1]=p; rgb[2]=q; break;
    }
}

struct Node {
    float x, y, z;
    float amp, fade;
    float freq, db, ageS;
    int   slot, frame, harm;
    float r, g, b;
};

static void pushEdge(std::vector<float>& buf, int& count,
                     const Node& a, const Node& b,
                     const float* rgb, float width, float alpha) {
    buf.push_back(a.x); buf.push_back(a.y); buf.push_back(a.z);
    buf.push_back(b.x); buf.push_back(b.y); buf.push_back(b.z);
    buf.push_back(width);
    buf.push_back(rgb[0]); buf.push_back(rgb[1]); buf.push_back(rgb[2]);
    buf.push_back(alpha);
    ++count;
}

// MST sobre grafo kNN. n<=~400, fuerza bruta es mas rapido que armar un kd-tree.
static void mstEdges(const std::vector<float>& px, const std::vector<float>& py,
                     int n, int knn, const std::vector<std::pair<int,int>>& exclude,
                     std::vector<std::pair<int,int>>& out) {
    out.clear();
    if (n < 2) return;
    // Prim sobre distancias euclidianas en el espacio 2D dado
    std::vector<float> best(n, 1e30f);
    std::vector<int> parent(n, -1), used(n, 0);
    best[0] = 0.0f;
    for (int it = 0; it < n; ++it) {
        int u = -1; float bv = 1e30f;
        for (int i = 0; i < n; ++i) if (!used[i] && best[i] < bv) { bv = best[i]; u = i; }
        if (u < 0) break;
        used[u] = 1;
        if (parent[u] >= 0) {
            int a = std::min(u, parent[u]), b = std::max(u, parent[u]);
            bool skip = false;
            for (auto& e : exclude)
                if (std::min(e.first, e.second) == a && std::max(e.first, e.second) == b) {
                    skip = true; break;
                }
            if (!skip) out.emplace_back(a, b);
        }
        for (int v = 0; v < n; ++v) {
            if (used[v]) continue;
            float dx = px[u] - px[v], dy = py[u] - py[v];
            float d = dx * dx + dy * dy;
            if (d < best[v]) { best[v] = d; parent[v] = u; }
        }
    }
    (void)knn;
}

void buildGraph(const Analysis& A, int frameIdx, const GraphParams& gp, GraphOut& out) {
    out.nodes.clear(); out.edgesWeb.clear(); out.edgesPartial.clear(); out.edgesHarm.clear();
    out.labelPos.clear(); out.labelData.clear();
    out.nNodes = out.nWeb = out.nPartial = out.nHarm = out.nLabels = 0;
    if (A.nFrames <= 0) return;

    const int j = std::max(0, std::min(frameIdx, A.nFrames - 1));
    const float sr = (float)A.p.sr, hop = (float)A.p.hop;
    const float flux = A.flux.empty() ? 0.0f :
        A.flux[j] / (*std::max_element(A.flux.begin(), A.flux.end()) + 1e-9f);
    const float pulse = 1.0f + gp.fluxPulse * flux;

    std::vector<Node> nodes;
    std::vector<std::pair<int,int>> ePart, eHarm, eWeb;
    std::vector<float> wPart, wHarm;
    int hot = 0;

    const bool manifold = (gp.layout == MANIFOLD);
    const float halfLife = manifold ? gp.fadeHalfMan : gp.fadeHalf;
    // mas alla del cutoff el nodo es invisible: ni se construye
    const float maxAge = halfLife * std::log2(1.0f / std::max(gp.alphaCutoff, 1e-4f));

    if (manifold) {
        // ---- nodos = cuadros proyectados por PCA de sus MFCC
        int i0 = std::max(0, j - (int)(std::min(gp.trailSec, maxAge) * sr / hop) + 1);
        for (int f = i0; f <= j; f += gp.nodeHop) {
            if (!A.active[f]) continue;
            float ageS = (j - f) * hop / sr;
            if (ageS > maxAge) continue;
            Node nd{};
            nd.x = A.pca[(size_t)f * 3 + 0] * pulse;
            nd.y = A.pca[(size_t)f * 3 + 1] * pulse;
            nd.z = A.pca[(size_t)f * 3 + 2] * pulse;
            nd.frame = f; nd.ageS = ageS;
            nd.fade = std::pow(0.5f, ageS / std::max(halfLife, 1e-3f));
            nd.freq = A.centroid[f]; nd.db = A.tonality[f];
            nd.harm = A.pitchClass[f];
            nodes.push_back(nd);
        }
        if (nodes.size() < 4) return;
        // amplitud: mezcla de energia y tonalidad
        float rmsMax = 1e-9f;
        for (auto& nd : nodes) rmsMax = std::max(rmsMax, A.rms[nd.frame]);
        for (auto& nd : nodes) {
            float a = A.rms[nd.frame] / rmsMax;
            nd.amp = std::min(1.0f, std::max(0.05f, a) * 0.4f + 0.6f * A.conc[nd.frame]);
            if (gp.colorChroma) chromaRgb(A.pitchClass[nd.frame], &nd.r);
            else { nd.r = C_PARTIAL[0]; nd.g = C_PARTIAL[1]; nd.b = C_PARTIAL[2]; }
        }
        hot = (int)nodes.size() - 1;

        // (1) trayectoria temporal, sin saltar silencios
        for (size_t i = 0; i + 1 < nodes.size(); ++i)
            if (nodes[i + 1].frame - nodes[i].frame <= gp.nodeHop * 3) {
                ePart.emplace_back((int)i, (int)i + 1);
                wPart.push_back(std::min(nodes[i].amp, nodes[i + 1].amp));
            }

        // (2) vecinos en el manifold: cerca en timbre pero lejos en el tiempo.
        //     estas son las cuerdas que marcan motivos repetidos.
        const int n = (int)nodes.size();
        std::vector<float> dists;
        dists.reserve((size_t)n * 3);
        std::vector<std::pair<float,int>> cand;
        for (int i = 0; i < n; ++i) {
            cand.clear();
            for (int k = 0; k < n; ++k) {
                if (std::abs(k - i) <= 6) continue;
                float dx = nodes[i].x - nodes[k].x, dy = nodes[i].y - nodes[k].y,
                      dz = nodes[i].z - nodes[k].z;
                cand.emplace_back(std::sqrt(dx*dx + dy*dy + dz*dz), k);
            }
            if (cand.empty()) continue;
            std::partial_sort(cand.begin(),
                              cand.begin() + std::min((size_t)2, cand.size()), cand.end());
            for (int m = 0; m < 2 && m < (int)cand.size(); ++m) dists.push_back(cand[m].first);
        }
        float thr = 0.0f;
        if (!dists.empty()) {
            size_t kk = (size_t)(0.45f * (dists.size() - 1));
            std::nth_element(dists.begin(), dists.begin() + kk, dists.end());
            thr = dists[kk];
        }
        for (int i = 0; i < n; ++i) {
            cand.clear();
            for (int k = 0; k < n; ++k) {
                if (std::abs(k - i) <= 6) continue;
                float dx = nodes[i].x - nodes[k].x, dy = nodes[i].y - nodes[k].y,
                      dz = nodes[i].z - nodes[k].z;
                float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (d < thr) cand.emplace_back(d, k);
            }
            std::sort(cand.begin(), cand.end());
            for (int m = 0; m < 2 && m < (int)cand.size(); ++m) {
                eHarm.emplace_back(i, cand[m].second);
                wHarm.push_back(1.0f - cand[m].first / (thr + 1e-9f));
            }
        }
        std::vector<float> px(n), py(n);
        for (int i = 0; i < n; ++i) { px[i] = nodes[i].x; py[i] = nodes[i].y; }
        mstEdges(px, py, n, gp.knn, ePart, eWeb);

    } else {
        // ---- nodos = picos espectrales dentro de la ventana
        int winFrames = std::max(6, (int)(std::min(gp.windowSec, maxAge) * sr / hop));
        int i0 = std::max(0, j - winFrames + 1);
        std::vector<int> frames;
        for (int f = i0; f <= j; f += gp.nodeHop) frames.push_back(f);
        int nSlots = std::max((int)frames.size() - 1, 1);

        float dMax = -1e9f;
        for (size_t s = 0; s < frames.size(); ++s) {
            int f = frames[s];
            for (int q = A.peakOffset[f]; q < A.peakOffset[f + 1]; ++q)
                dMax = std::max(dMax, A.peakDb[q]);
        }
        if (dMax < -1e8f) return;
        dMax = std::max(dMax, A.refDb - 20.0f);

        for (size_t s = 0; s < frames.size(); ++s) {
            int f = frames[s];
            for (int q = A.peakOffset[f]; q < A.peakOffset[f + 1]; ++q) {
                Node nd{};
                nd.freq = A.peakFreq[q]; nd.db = A.peakDb[q];
                nd.slot = (int)s; nd.frame = f;
                nd.amp = std::min(1.0f, std::max(0.02f, (nd.db - (dMax - 42.0f)) / 42.0f));
                nd.ageS = ((int)frames.size() - 1 - (int)s) * gp.nodeHop * hop / sr;
                nd.fade = std::pow(0.5f, nd.ageS / std::max(halfLife, 1e-3f));
                nd.harm = 0;
                nd.r = C_PARTIAL[0]; nd.g = C_PARTIAL[1]; nd.b = C_PARTIAL[2];
                nodes.push_back(nd);
            }
        }
        if (nodes.size() < 4) return;

        const float span = std::log2(A.p.fmax / A.p.fmin);
        const float spreadN = std::min(1.6f, std::max(0.25f, A.spread[j] / 4000.0f));
        for (auto& nd : nodes) {
            float fc = std::min(A.p.fmax, std::max(A.p.fmin, nd.freq));
            float oc = std::log2(fc / A.p.fmin) / span;                 // octava normalizada
            float chr = std::fmod(std::log2(fc / A.p.fmin), 1.0f);      // clase de altura
            float tRel = (float)nd.slot / nSlots - 1.0f;                // -1 viejo, 0 ahora
            switch (gp.layout) {
                case WALL:
                    nd.x = tRel * 2.0f + 1.0f; nd.y = oc * 2.0f - 1.0f;
                    nd.z = nd.amp * 1.3f - 0.65f; break;
                case SPHERE: {
                    float az = 2 * PI * chr, el = (oc - 0.5f) * PI * 0.9f;
                    float r = 0.45f + 0.55f * (1 + tRel) + 0.25f * nd.amp;
                    nd.x = r * std::cos(el) * std::cos(az);
                    nd.y = r * std::cos(el) * std::sin(az);
                    nd.z = r * std::sin(el); break;
                }
                case CIRCLE: {
                    // tiempo -> radio, frecuencia -> angulo (galaxia)
                    float ang = 2 * PI * oc;
                    float r = (0.12f + 1.05f * (1 + tRel)) * spreadN;
                    nd.x = r * std::cos(ang); nd.y = r * std::sin(ang);
                    nd.z = (nd.amp - 0.5f) * 0.55f; break;
                }
                default: {   // HELIX
                    float ang = 2 * PI * chr;
                    float r = 0.22f + 0.78f * oc + 0.12f * nd.amp;
                    nd.x = r * std::cos(ang); nd.y = r * std::sin(ang);
                    nd.z = tRel * 2.7f + 1.35f; break;
                }
            }
            nd.x *= pulse; nd.y *= pulse; nd.z *= pulse;
        }

        // partial tracking: unir picos de cuadros consecutivos con frecuencia similar
        const float tol = gp.tolSemis / 12.0f;
        std::vector<std::vector<int>> bySlot(frames.size());
        for (size_t i = 0; i < nodes.size(); ++i) bySlot[nodes[i].slot].push_back((int)i);
        for (size_t s = 0; s + 1 < frames.size(); ++s) {
            auto& Aidx = bySlot[s]; auto& Bidx = bySlot[s + 1];
            if (Aidx.empty() || Bidx.empty()) continue;
            std::vector<int> order(Aidx.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return nodes[Aidx[a]].amp > nodes[Aidx[b]].amp; });
            std::vector<char> used(Bidx.size(), 0);
            for (int oi : order) {
                int ai = Aidx[oi]; int bestB = -1; float bestC = 1e9f;
                for (size_t k = 0; k < Bidx.size(); ++k) {
                    if (used[k]) continue;
                    float c = std::fabs(std::log2(nodes[ai].freq / nodes[Bidx[k]].freq));
                    if (c < bestC) { bestC = c; bestB = (int)k; }
                }
                if (bestB >= 0 && bestC < tol) {
                    used[bestB] = 1;
                    ePart.emplace_back(ai, Bidx[bestB]);
                    wPart.push_back(std::min(nodes[ai].amp, nodes[Bidx[bestB]].amp));
                }
            }
        }

        // abanico armonico desde el pico dominante
        hot = 0;
        for (size_t i = 1; i < nodes.size(); ++i) if (nodes[i].db > nodes[hot].db) hot = (int)i;
        float f0 = nodes[hot].freq;
        for (int i : bySlot[nodes[hot].slot]) {
            if (i == hot) continue;
            float r = nodes[i].freq / f0;
            int nh = (int)std::lround(r);
            if (nh >= 1 && std::fabs(r - nh) < 0.06f) {
                nodes[i].harm = nh;
                eHarm.emplace_back(hot, i);
                wHarm.push_back(nodes[i].amp);
            }
        }

        std::vector<float> px(nodes.size()), py(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            px[i] = (float)nodes[i].slot / nSlots;
            float fc = std::min(A.p.fmax, std::max(A.p.fmin, nodes[i].freq));
            py[i] = std::log2(fc / A.p.fmin) / span;
        }
        mstEdges(px, py, (int)nodes.size(), gp.knn, ePart, eWeb);
    }

    // ------------------------------------------------- color por rol
    const int n = (int)nodes.size();
    std::vector<int> deg(n, 0);
    for (auto& e : ePart) { deg[e.first]++; deg[e.second]++; }
    for (auto& e : eHarm) { deg[e.first]++; deg[e.second]++; }
    for (auto& e : eWeb)  { deg[e.first]++; deg[e.second]++; }
    if (!(manifold && gp.colorChroma)) {
        for (int i = 0; i < n; ++i) {
            if (nodes[i].amp > 0.6f) { nodes[i].r=C_HOT[0]; nodes[i].g=C_HOT[1]; nodes[i].b=C_HOT[2]; }
            if (deg[i] <= 1)         { nodes[i].r=C_ISO[0]; nodes[i].g=C_ISO[1]; nodes[i].b=C_ISO[2]; }
        }
    }
    nodes[hot].r = C_HOT[0]; nodes[hot].g = C_HOT[1]; nodes[hot].b = C_HOT[2];

    // ------------------------------------------------- volcar buffers
    out.nodes.reserve((size_t)n * 8);
    for (int i = 0; i < n; ++i) {
        const Node& nd = nodes[i];
        float size = 2.5f + 9.0f * nd.amp;
        if (gp.shrinkFade) size *= (0.30f + 0.70f * nd.fade);
        if (i == hot) size *= 2.2f;
        float alpha = std::min(1.0f, 0.30f + 0.70f * nd.amp) * nd.fade;
        out.nodes.push_back(nd.x); out.nodes.push_back(nd.y); out.nodes.push_back(nd.z);
        out.nodes.push_back(size);
        out.nodes.push_back(nd.r); out.nodes.push_back(nd.g); out.nodes.push_back(nd.b);
        out.nodes.push_back(alpha);
    }
    out.nNodes = n;
    out.hotIndex = hot;

    for (auto& e : eWeb) {
        float a = std::min(nodes[e.first].fade, nodes[e.second].fade) * 0.22f;
        pushEdge(out.edgesWeb, out.nWeb, nodes[e.first], nodes[e.second], C_WEB, 0.35f, a);
    }
    for (size_t k = 0; k < ePart.size(); ++k) {
        auto& e = ePart[k];
        float w = std::min(1.0f, std::max(0.0f, wPart[k]));
        float a = std::min(1.0f, 0.25f + 0.75f * w) *
                  std::min(nodes[e.first].fade, nodes[e.second].fade);
        const float* rgb = (manifold && gp.colorChroma) ? &nodes[e.first].r : C_PARTIAL;
        pushEdge(out.edgesPartial, out.nPartial, nodes[e.first], nodes[e.second],
                 rgb, 0.35f + 1.1f * w, a);
    }
    for (size_t k = 0; k < eHarm.size(); ++k) {
        auto& e = eHarm[k];
        float w = std::min(1.0f, std::max(0.0f, wHarm[k]));
        float a = std::min(1.0f, 0.30f + 0.70f * w) *
                  std::min(nodes[e.first].fade, nodes[e.second].fade);
        pushEdge(out.edgesHarm, out.nHarm, nodes[e.first], nodes[e.second],
                 C_HARM, 0.40f + 0.9f * w, a);
    }

    // ------------------------------------------------- etiquetas
    // los mas fuertes, pero separados entre si para que las cajas no se amontonen
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return nodes[a].amp * nodes[a].fade > nodes[b].amp * nodes[b].fade; });
    std::vector<int> chosen{ hot };
    for (int i : order) {
        if ((int)chosen.size() >= gp.maxLabels) break;
        if (i == hot) continue;
        float minD = 1e9f;
        for (int c : chosen) {
            float dx = nodes[i].x - nodes[c].x, dy = nodes[i].y - nodes[c].y,
                  dz = nodes[i].z - nodes[c].z;
            minD = std::min(minD, std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        if (minD > gp.labelMinSep) chosen.push_back(i);
    }
    for (int i : chosen) {
        const Node& nd = nodes[i];
        out.labelPos.push_back(nd.x); out.labelPos.push_back(nd.y); out.labelPos.push_back(nd.z);
        out.labelPos.push_back(nd.r); out.labelPos.push_back(nd.g); out.labelPos.push_back(nd.b);
        out.labelPos.push_back(std::min(1.0f, nd.fade * 1.4f));
        out.labelData.push_back(nd.freq);
        out.labelData.push_back(nd.db);
        out.labelData.push_back(nd.amp);
        out.labelData.push_back(nd.ageS * 1000.0f);
        out.labelData.push_back((float)nd.harm);
        out.labelData.push_back((float)nd.frame);
        out.labelData.push_back(manifold ? A.conc[nd.frame] : 0.0f);
        out.labelData.push_back(i == hot ? 1.0f : 0.0f);
        ++out.nLabels;
    }
}

} // namespace bg
