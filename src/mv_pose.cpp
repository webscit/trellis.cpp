#include "mv_pose.h"

#include <algorithm>
#include <cmath>

namespace trellis {

namespace {

// Rescale the box [x0, x0+w) x [y0, y0+h) of a source plane to the kSilSize square: the longer
// side fills it, the shorter one is centred. `inside(sx, sy)` samples the source at a point.
template <class F>
Silhouette normalise(double x0, double y0, double w, double h, F inside) {
    Silhouette s((size_t)kSilSize * kSilSize, 0);
    if (!(w > 0) || !(h > 0)) return s;
    const double side = std::max(w, h);
    const double ox = x0 - (side - w) / 2, oy = y0 - (side - h) / 2;
    for (int i = 0; i < kSilSize; ++i)
        for (int j = 0; j < kSilSize; ++j) {
            const double sx = ox + (j + 0.5) / kSilSize * side;
            const double sy = oy + (i + 0.5) / kSilSize * side;
            s[(size_t)i * kSilSize + j] = inside(sx, sy) ? 1 : 0;
        }
    return s;
}

float wrap360(float a) {
    a = std::fmod(a, 360.0f);
    return a < 0 ? a + 360.0f : a;
}

float ang_dist(float a, float b) {
    const float d = std::fabs(wrap360(a) - wrap360(b));
    return std::min(d, 360.0f - d);
}

}  // namespace

Silhouette matte_silhouette(const unsigned char* rgba, int w, int h) {
    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (rgba[((size_t)y * w + x) * 4 + 3] > 127) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
    if (x1 < 0) return Silhouette((size_t)kSilSize * kSilSize, 0);
    return normalise(x0, y0, x1 + 1 - x0, y1 + 1 - y0, [&](double sx, double sy) {
        const int x = (int)std::floor(sx), y = (int)std::floor(sy);
        if (x < 0 || y < 0 || x >= w || y >= h) return false;
        return rgba[((size_t)y * w + x) * 4 + 3] > 127;
    });
}

Silhouette voxel_silhouette(const std::vector<std::array<int,3>>& cells, int G,
                            float az_deg, float el_deg, const BlendParams& bp) {
    if (cells.empty() || G <= 0) return Silhouette((size_t)kSilSize * kSilSize, 0);
    // Camera basis on the grid axes: right r = direction of (az + 90, 0), up u = (az, el + 90).
    const auto r = view_dir_grid(az_deg + 90.0f, 0.0f, bp);
    const auto u = view_dir_grid(az_deg, el_deg + 90.0f, bp);
    const double cell = 2.0 / G;
    // Half footprint of one cube along r and u (its projected bounding box).
    const double er = 0.5 * cell * (std::fabs(r[0]) + std::fabs(r[1]) + std::fabs(r[2]));
    const double eu = 0.5 * cell * (std::fabs(u[0]) + std::fabs(u[1]) + std::fabs(u[2]));
    std::vector<double> ps(cells.size()), pt(cells.size());
    double smin = 1e9, smax = -1e9, tmin = 1e9, tmax = -1e9;
    for (size_t n = 0; n < cells.size(); ++n) {
        double p[3];
        for (int a = 0; a < 3; ++a) p[a] = (cells[n][a] + 0.5) * cell - 1.0;
        ps[n] = p[0] * r[0] + p[1] * r[1] + p[2] * r[2];
        pt[n] = -(p[0] * u[0] + p[1] * u[1] + p[2] * u[2]);   // image rows grow downward
        smin = std::min(smin, ps[n]); smax = std::max(smax, ps[n]);
        tmin = std::min(tmin, pt[n]); tmax = std::max(tmax, pt[n]);
    }
    smin -= er; smax += er; tmin -= eu; tmax += eu;
    // Rasterise the footprints at kSilSize^2 over the bbox, then reuse `normalise` as a lookup.
    const double w = smax - smin, h = tmax - tmin, side = std::max(w, h);
    const double ox = smin - (side - w) / 2, oy = tmin - (side - h) / 2;
    const double px = side / kSilSize;
    Silhouette s((size_t)kSilSize * kSilSize, 0);
    for (size_t n = 0; n < cells.size(); ++n) {
        const int j0 = std::max(0, (int)std::ceil((ps[n] - er - ox) / px - 0.5));
        const int j1 = std::min(kSilSize - 1, (int)std::floor((ps[n] + er - ox) / px - 0.5));
        const int i0 = std::max(0, (int)std::ceil((pt[n] - eu - oy) / px - 0.5));
        const int i1 = std::min(kSilSize - 1, (int)std::floor((pt[n] + eu - oy) / px - 0.5));
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j) s[(size_t)i * kSilSize + j] = 1;
    }
    return s;
}

float silhouette_iou(const Silhouette& a, const Silhouette& b) {
    size_t in = 0, un = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { in += a[i] & b[i]; un += a[i] | b[i]; }
    return un ? (float)in / (float)un : 0.0f;
}

std::vector<std::array<int,3>> occupied_cells(const std::vector<float>& logits, int S) {
    std::vector<std::array<int,3>> out;
    for (int x = 0; x < S; ++x)
        for (int y = 0; y < S; ++y)
            for (int z = 0; z < S; ++z)
                if (logits[((size_t)x * S + y) * S + z] > 0.0f) out.push_back({x, y, z});
    return out;
}

float PoseTable::at(int v, float az) const {
    const int K = (int)iou[(size_t)v].size();
    const int k = (int)std::lround(wrap360(az) / step) % K;
    return iou[(size_t)v][(size_t)k];
}

PoseTable pose_table(const std::vector<std::array<int,3>>& cells, int G,
                     const std::vector<ViewSpec>& views, const std::vector<Silhouette>& mattes,
                     FrontAxis front, int step) {
    PoseTable t;
    t.step = step;
    const int K = 360 / step;
    BlendParams bp;
    bp.front = front;
    t.iou.assign(views.size(), std::vector<float>((size_t)K, 0.0f));
    for (size_t v = 0; v < views.size(); ++v)
        for (int k = 0; k < K; ++k)
            t.iou[v][(size_t)k] = silhouette_iou(
                mattes[v], voxel_silhouette(cells, G, (float)(k * step), views[v].el_deg, bp));
    return t;
}

YawChoice choose_yaw(const PoseTable& t, const std::vector<ViewSpec>& views,
                     const std::vector<bool>& known, float margin) {
    YawChoice c;
    int best = 0;
    for (int k = 0; k < 4; ++k) {
        double s = 0;
        int n = 0;
        for (size_t v = 0; v < views.size(); ++v)
            if (known[v]) { s += t.at((int)v, views[v].az_deg + 90.0f * k); ++n; }
        c.scores.push_back(n ? (float)(s / n) : 0.0f);
        if (c.scores[(size_t)k] > c.scores[(size_t)best]) best = k;
    }
    c.score0 = c.scores[0];
    c.best = c.scores[(size_t)best];
    if (c.best - c.score0 < margin) best = 0;   // keep the supplied frame unless clearly beaten
    c.yaw = 90.0f * best;
    for (int k = 0; k < 4; ++k)
        if (k != best && c.scores[(size_t)k] >= c.scores[(size_t)best] - margin) c.ambiguous = true;
    return c;
}

AzChoice choose_azimuth(const PoseTable& t, int v, float yaw, const std::vector<float>& other_az,
                        float margin) {
    const std::vector<float>& row = t.iou[(size_t)v];
    const int K = (int)row.size();
    float best = 0;
    for (float x : row) best = std::max(best, x);
    // Local maxima (circular) within `margin` of the best: the distinct hypotheses.
    std::vector<int> cand;
    for (int k = 0; k < K; ++k) {
        const float l = row[(size_t)((k + K - 1) % K)], r = row[(size_t)((k + 1) % K)];
        if (row[(size_t)k] >= l && row[(size_t)k] >= r && row[(size_t)k] >= best - margin) cand.push_back(k);
    }
    AzChoice c;
    float best_sep = -1;
    for (int k : cand) {
        const float az = wrap360(k * t.step - yaw);   // back to the views' own frame
        c.alternatives.push_back(az);
        float sep = 360;
        for (float o : other_az) sep = std::min(sep, ang_dist(az, o));
        if (other_az.empty()) sep = 0;
        if (sep > best_sep + 1e-3f || (std::fabs(sep - best_sep) <= 1e-3f && row[(size_t)k] > c.iou)) {
            best_sep = sep; c.az = az; c.iou = row[(size_t)k];
        }
    }
    return c;
}

std::vector<size_t> front_fill(const std::vector<std::array<int,3>>& blended,
                               const std::vector<std::array<int,3>>& front, int G,
                               float az_deg, float el_deg, const BlendParams& bp) {
    const auto r = view_dir_grid(az_deg + 90.0f, 0.0f, bp);
    const auto u = view_dir_grid(az_deg, el_deg + 90.0f, bp);
    auto key = [&](const std::array<int,3>& c) {
        double s = 0, t = 0;
        for (int a = 0; a < 3; ++a) { const double p = c[a] + 0.5; s += p * r[a]; t += p * u[a]; }
        const long long i = (long long)std::floor(s), j = (long long)std::floor(t);
        return (i + 4LL * G) * (8LL * G) + (j + 4LL * G);
    };
    std::vector<long long> cov;
    cov.reserve(blended.size());
    for (const auto& c : blended) cov.push_back(key(c));
    std::sort(cov.begin(), cov.end());
    std::vector<size_t> add;
    for (size_t n = 0; n < front.size(); ++n)
        if (!std::binary_search(cov.begin(), cov.end(), key(front[n]))) add.push_back(n);
    return add;
}

std::string silhouette_diff_ascii(const Silhouette& matte, const Silhouette& vox, int every) {
    std::string s;
    for (int i = 0; i < kSilSize; i += every) {
        for (int j = 0; j < kSilSize; j += every) {
            const size_t k = (size_t)i * kSilSize + j;
            s += matte[k] && vox[k] ? '#' : matte[k] ? 'm' : vox[k] ? 'v' : '.';
        }
        s += '\n';
    }
    return s;
}

}  // namespace trellis
