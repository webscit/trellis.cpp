// Printability report (docs/spec/32-print-ready.md §2.6): what a slicer and a printer will
// care about, measured on the final mesh in millimetres.
#include "mesh_print.h"
#include "tri_bvh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

namespace trellis {

namespace {

struct RayQuery {
    const float* verts;
    const int32_t* faces;
    int32_t skip;
    double origin[3], dir[3];
    double nearest;
    double hit_normal[3];   // unnormalised normal of the nearest face hit
};

// Moeller-Trumbore; keeps the nearest hit in front of the origin.
bool ray_visit(void* context, int32_t face) {
    RayQuery& q = *static_cast<RayQuery*>(context);
    if (face == q.skip) return true;
    const float *a = q.verts + 3 * (int64_t)q.faces[3 * (int64_t)face], *b = q.verts + 3 * (int64_t)q.faces[3 * (int64_t)face + 1],
                *c = q.verts + 3 * (int64_t)q.faces[3 * (int64_t)face + 2];
    const double e1[3] = {(double)b[0] - a[0], (double)b[1] - a[1], (double)b[2] - a[2]};
    const double e2[3] = {(double)c[0] - a[0], (double)c[1] - a[1], (double)c[2] - a[2]};
    const double p[3] = {q.dir[1] * e2[2] - q.dir[2] * e2[1], q.dir[2] * e2[0] - q.dir[0] * e2[2], q.dir[0] * e2[1] - q.dir[1] * e2[0]};
    const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (std::fabs(det) < 1e-30) return true;
    const double s[3] = {q.origin[0] - a[0], q.origin[1] - a[1], q.origin[2] - a[2]};
    const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
    if (u < 0 || u > 1) return true;
    const double r[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
    const double v = (q.dir[0] * r[0] + q.dir[1] * r[1] + q.dir[2] * r[2]) / det;
    if (v < 0 || u + v > 1) return true;
    const double t = (e2[0] * r[0] + e2[1] * r[1] + e2[2] * r[2]) / det;
    if (t > 0 && t < q.nearest) {
        q.nearest = t;
        q.hit_normal[0] = e1[1] * e2[2] - e1[2] * e2[1];
        q.hit_normal[1] = e1[2] * e2[0] - e1[0] * e2[2];
        q.hit_normal[2] = e1[0] * e2[1] - e1[1] * e2[0];
    }
    return true;
}

}  // namespace

PrintReport print_report(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                         float size_mm, int up_axis, float wall_mm) {
    PrintReport r;
    r.audit = audit_mesh(verts, V, faces, F);
    r.wall_target_mm = wall_mm > 0 ? wall_mm : 0.8f;
    if (V <= 0 || F <= 0) { r.warnings.push_back("empty mesh"); return r; }

    const MeshAudit& a = r.audit;
    const float extent = std::max(a.bbox_max[0] - a.bbox_min[0], std::max(a.bbox_max[1] - a.bbox_min[1], a.bbox_max[2] - a.bbox_min[2]));
    const double mm = size_mm > 0 && extent > 0 ? size_mm / extent : 1.0;   // model units -> mm
    const int up = up_axis >= 0 && up_axis < 3 ? up_axis : 2, ax = (up + 1) % 3, ay = (up + 2) % 3;
    r.size_mm[0] = (float)((a.bbox_max[ax] - a.bbox_min[ax]) * mm);
    r.size_mm[1] = (float)((a.bbox_max[ay] - a.bbox_min[ay]) * mm);
    r.size_mm[2] = (float)((a.bbox_max[up] - a.bbox_min[up]) * mm);
    r.volume_cm3 = a.volume * mm * mm * mm / 1000.0;
    r.area_cm2 = a.area * mm * mm / 100.0;

    // ---- wall thickness: a ray from each face straight into the material ----
    const TriBvh bvh = TriBvh::build(verts, V, faces, F);
    const double reach = 4.0 * r.wall_target_mm / mm;   // model units; thicker than this is "thick"
    std::vector<float> thickness((size_t)F, -1.0f), face_area((size_t)F, 0.0f);
    std::vector<float> normals((size_t)F * 3, 0.0f);
    {
        const int nt = (int)std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> workers;
        const int64_t chunk = (F + nt - 1) / nt;
        for (int64_t begin = 0; begin < F; begin += chunk)
            workers.emplace_back([&, begin]() {
                for (int64_t f = begin; f < std::min(F, begin + chunk); ++f) {
                    const float *p0 = verts + 3 * (int64_t)faces[3 * f], *p1 = verts + 3 * (int64_t)faces[3 * f + 1], *p2 = verts + 3 * (int64_t)faces[3 * f + 2];
                    const double u[3] = {(double)p1[0] - p0[0], (double)p1[1] - p0[1], (double)p1[2] - p0[2]};
                    const double w[3] = {(double)p2[0] - p0[0], (double)p2[1] - p0[1], (double)p2[2] - p0[2]};
                    double nrm[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
                    const double len = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
                    if (len <= 0) continue;
                    face_area[(size_t)f] = (float)(0.5 * len);
                    RayQuery q{verts, faces, (int32_t)f, {0, 0, 0}, {0, 0, 0}, reach, {0, 0, 0}};
                    float lo[3], hi[3];
                    for (int k = 0; k < 3; ++k) {
                        nrm[k] /= len;
                        normals[3 * (size_t)f + k] = (float)nrm[k];
                        q.origin[k] = ((double)p0[k] + p1[k] + p2[k]) / 3.0;
                        q.dir[k] = -nrm[k];
                        const double end = q.origin[k] + q.dir[k] * reach;
                        lo[k] = (float)std::min(q.origin[k], end) - 1e-6f;
                        hi[k] = (float)std::max(q.origin[k], end) + 1e-6f;
                    }
                    bvh.visit_overlapping(lo, hi, ray_visit, &q);
                    // A wall has two roughly opposite sides. Next to an ordinary convex edge
                    // the ray leaves through the adjacent face instead, which is at an angle:
                    // that is a corner, not a thin wall.
                    if (q.nearest < reach) {
                        const double* h = q.hit_normal;
                        const double hl = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
                        const double facing = hl > 0 ? (h[0] * nrm[0] + h[1] * nrm[1] + h[2] * nrm[2]) / hl : 0.0;
                        if (facing > -0.5) q.nearest = reach;
                    }
                    thickness[(size_t)f] = (float)(q.nearest * mm);
                }
            });
        for (auto& w : workers) w.join();
    }
    {
        std::vector<std::pair<float, float>> sorted;   // (thickness, area)
        sorted.reserve((size_t)F);
        double total = 0, thin = 0;
        for (int64_t f = 0; f < F; ++f) {
            if (thickness[(size_t)f] < 0) continue;
            sorted.emplace_back(thickness[(size_t)f], face_area[(size_t)f]);
            total += face_area[(size_t)f];
            if (thickness[(size_t)f] < r.wall_target_mm) thin += face_area[(size_t)f];
        }
        std::sort(sorted.begin(), sorted.end());
        double running = 0;
        r.wall_p1_mm = sorted.empty() ? 0.0f : sorted.back().first;
        for (const auto& s : sorted) {
            running += s.second;
            if (running >= 0.01 * total) { r.wall_p1_mm = s.first; break; }
        }
        r.thin_area_pct = total > 0 ? 100.0 * thin / total : 0.0;
        r.wall_capped = r.wall_p1_mm >= (float)(reach * mm) - 1e-4f;
    }

    // ---- stance: does the centre of mass sit over the contact patch? ----
    const double tolerance = 0.1 / mm;   // 0.1 mm: what counts as touching the plate
    const double floor_level = a.bbox_min[up];
    double com[3] = {0, 0, 0}, volume6 = 0;
    double down_area = 0, total_area = 0;
    for (int64_t f = 0; f < F; ++f) {
        const float *p0 = verts + 3 * (int64_t)faces[3 * f], *p1 = verts + 3 * (int64_t)faces[3 * f + 1], *p2 = verts + 3 * (int64_t)faces[3 * f + 2];
        const double v6 = (double)p0[0] * ((double)p1[1] * p2[2] - (double)p1[2] * p2[1]) - (double)p0[1] * ((double)p1[0] * p2[2] - (double)p1[2] * p2[0]) +
                          (double)p0[2] * ((double)p1[0] * p2[1] - (double)p1[1] * p2[0]);
        volume6 += v6;
        for (int k = 0; k < 3; ++k) com[k] += v6 * ((double)p0[k] + p1[k] + p2[k]) / 4.0;
        // Overhang: faces looking down steeper than 45 degrees, other than the base itself.
        total_area += face_area[(size_t)f];
        const double height = ((double)p0[up] + p1[up] + p2[up]) / 3.0 - floor_level;
        if (normals[3 * (size_t)f + up] < -0.70710678f && height > tolerance) down_area += face_area[(size_t)f];
    }
    r.overhang_pct = total_area > 0 ? 100.0 * down_area / total_area : 0.0;
    if (std::fabs(volume6) > 0) for (double& c : com) c /= volume6;

    std::vector<std::pair<double, double>> contact;
    for (int64_t v = 0; v < V; ++v)
        if (verts[3 * v + up] - floor_level <= tolerance) contact.emplace_back(verts[3 * v + ax], verts[3 * v + ay]);
    std::sort(contact.begin(), contact.end());
    contact.erase(std::unique(contact.begin(), contact.end()), contact.end());
    std::vector<std::pair<double, double>> hull(2 * contact.size() + 1);
    size_t h = 0;
    auto cross = [](const std::pair<double, double>& o, const std::pair<double, double>& p, const std::pair<double, double>& q) {
        return (p.first - o.first) * (q.second - o.second) - (p.second - o.second) * (q.first - o.first);
    };
    if (contact.size() >= 3) {
        for (size_t i = 0; i < contact.size(); ++i) { while (h >= 2 && cross(hull[h-2], hull[h-1], contact[i]) <= 0) --h; hull[h++] = contact[i]; }
        for (size_t i = contact.size() - 1, lower = h + 1; i-- > 0;) { while (h >= lower && cross(hull[h-2], hull[h-1], contact[i]) <= 0) --h; hull[h++] = contact[i]; }
    }
    double hull_area = 0, margin = 1e30;
    for (size_t i = 0; i + 1 < h; ++i) {
        hull_area += 0.5 * (hull[i].first * hull[i+1].second - hull[i+1].first * hull[i].second);
        const double ex = hull[i+1].first - hull[i].first, ey = hull[i+1].second - hull[i].second, len = std::sqrt(ex * ex + ey * ey);
        if (len > 0) margin = std::min(margin, (ex * (com[ay] - hull[i].second) - ey * (com[ax] - hull[i].first)) / len);
    }
    r.base_area_mm2 = (float)(hull_area * mm * mm);
    r.flat_base = h >= 4 && r.base_area_mm2 >= 0.02f * r.size_mm[0] * r.size_mm[1];
    if (h >= 4) {
        r.tip_angle_deg = (float)(std::atan2(margin, std::max(1e-9, com[up] - floor_level)) * 57.29577951308232);
        r.stands = margin > 0;
    }

    // ---- verdict ----
    char line[256];
    if (!a.watertight()) r.warnings.push_back("not watertight: " + audit_summary(a));
    if (a.components > 1) {
        std::snprintf(line, sizeof line, "%d separate shells: unattached parts print in mid-air", a.components);
        r.warnings.push_back(line);
    }
    if (r.wall_p1_mm < 0.75f * r.wall_target_mm || r.thin_area_pct > 10.0) {
        std::snprintf(line, sizeof line, "%.1f%% of the surface is thinner than %.2f mm (thinnest 1%%: %.2f mm); raise --print-wall or --print-size",
                      r.thin_area_pct, r.wall_target_mm, r.wall_p1_mm);
        r.warnings.push_back(line);
    }
    if (!r.flat_base) r.warnings.push_back("no flat base: the model touches the plate on points; use --print-base");
    else if (!r.stands) r.warnings.push_back("the centre of mass is outside the base: it will fall over; use --print-base auto");
    else if (r.tip_angle_deg < 10.0f) {
        std::snprintf(line, sizeof line, "tips over at %.1f degrees of tilt; use --print-base auto for a wider base", r.tip_angle_deg);
        r.warnings.push_back(line);
    }
    if (r.overhang_pct > 25.0) {
        std::snprintf(line, sizeof line, "%.0f%% of the surface overhangs more than 45 degrees: expect a lot of support material", r.overhang_pct);
        r.warnings.push_back(line);
    }
    return r;
}

std::string print_report_text(const PrintReport& r) {
    char buffer[1024];
    std::snprintf(buffer, sizeof buffer,
        "  print report: %s\n"
        "    size        %.1f x %.1f x %.1f mm, volume %.1f cm3, surface %.1f cm2\n"
        "    watertight  %s (%d shell%s)\n"
        "    walls       thinnest 1%% of the surface: %s%.2f mm; %.1f%% (incl. fine relief) under the %.2f mm target\n"
        "    base        %s, contact %.0f mm2; %s (tips over at %.1f deg)\n"
        "    overhangs   %.0f%% of the surface steeper than 45 deg\n",
        r.ok() ? "OK" : "WARNINGS", r.size_mm[0], r.size_mm[1], r.size_mm[2], r.volume_cm3, r.area_cm2,
        r.audit.watertight() ? "yes" : "NO", r.audit.components, r.audit.components == 1 ? "" : "s",
        r.wall_capped ? ">= " : "", r.wall_p1_mm, r.thin_area_pct, r.wall_target_mm,
        r.flat_base ? "flat" : "NOT flat", r.base_area_mm2, r.stands ? "stands" : "FALLS OVER", r.tip_angle_deg,
        r.overhang_pct);
    std::string text = buffer;
    if (!r.palette.empty()) {
        std::snprintf(buffer, sizeof buffer, "    colours     %d filaments, %d patches:", (int)r.palette.size(), r.color_patches);
        text += buffer;
        for (size_t s = 0; s < r.palette.size(); ++s) {
            std::snprintf(buffer, sizeof buffer, "%s %d #%06X %.1f%%", s % 4 == 0 && s ? "\n               " : "",
                          (int)s + 1, (unsigned)r.palette[s], 100 * r.palette_share[s]);
            text += buffer;
        }
        text += "\n";
    }
    for (const std::string& w : r.warnings) text += "    warning     " + w + "\n";
    return text;
}

std::string print_report_json(const PrintReport& r) {
    auto escape = [](const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"' || c == '\\') { out += '\\'; out += c; }
            else if ((unsigned char)c < 0x20) out += ' ';
            else out += c;
        }
        return out;
    };
    char buffer[1024];
    std::snprintf(buffer, sizeof buffer,
        "{\"ok\":%s,\"size_mm\":[%.2f,%.2f,%.2f],\"volume_cm3\":%.3f,\"surface_cm2\":%.3f,"
        "\"watertight\":%s,\"shells\":%d,\"triangles\":%lld,"
        "\"wall\":{\"target_mm\":%.3f,\"thinnest_1pct_mm\":%.3f,\"at_least\":%s,\"thin_area_pct\":%.2f},"
        "\"base\":{\"flat\":%s,\"contact_mm2\":%.1f,\"stands\":%s,\"tip_angle_deg\":%.1f},"
        "\"overhang_pct\":%.1f,",
        r.ok() ? "true" : "false", r.size_mm[0], r.size_mm[1], r.size_mm[2], r.volume_cm3, r.area_cm2,
        r.audit.watertight() ? "true" : "false", r.audit.components, (long long)r.audit.faces,
        r.wall_target_mm, r.wall_p1_mm, r.wall_capped ? "true" : "false", r.thin_area_pct,
        r.flat_base ? "true" : "false", r.base_area_mm2, r.stands ? "true" : "false", r.tip_angle_deg, r.overhang_pct);
    std::string json = buffer;
    if (!r.palette.empty()) {
        json += "\"colors\":{\"patches\":" + std::to_string(r.color_patches) + ",\"palette\":[";
        for (size_t s = 0; s < r.palette.size(); ++s) {
            std::snprintf(buffer, sizeof buffer, "%s{\"slot\":%d,\"hex\":\"#%06X\",\"share_pct\":%.2f}", s ? "," : "",
                          (int)s + 1, (unsigned)r.palette[s], 100 * r.palette_share[s]);
            json += buffer;
        }
        json += "]},";
    }
    json += "\"warnings\":[";
    for (size_t i = 0; i < r.warnings.size(); ++i) json += (i ? ",\"" : "\"") + escape(r.warnings[i]) + "\"";
    return json + "]}\n";
}

}  // namespace trellis
