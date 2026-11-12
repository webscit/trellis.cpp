// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate    multipart/form-data with an "image" file part; optional text
//                      fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                      (threshold|birefnet), "uv" (xatlas = default, unique
//                      chart space; box = faster projection), "band" (narrow-band
//                      DC remesh band width, default 1 — see --band), "model"
//                      (trellis|pixal3d — one directory holds both), "faces"
//                      (QEM decimation face target, >= 1000 or "auto" — see
//                      --faces), "atlas" (UV atlas px, 256..8192 or "auto"),
//                      "webp" (on|off; off = PNG textures), and for
//                      pixal3d the camera: "fov" (degrees), "mesh_scale",
//                      "extend_pixel", "texture" (on|off; off = geometry
//                      only, like --no-texture), and "normal_map" (on|off;
//                      on = bake a tangent-space normal map, like --normal-map).
//                      Returns model/gltf-binary.
//
//                    Multi-view (docs/spec/30-multiview.md): send several "image" parts
//                    of the same object plus "azimuths" (comma list, one per image in
//                    the same order; degrees or front|back|left|right|top|bottom) and
//                    optionally "elevations" (default 0). "azimuths" is required with
//                    more than one image. Also "blend_temp", "front_axis"
//                    (-y|+y|+x|-x|ref-z|ref-x) and "yaw_offset". One image and no angle
//                    fields is exactly the single-image request.
//   POST /refine      multipart/form-data with a "mesh" file part (GLB or OBJ, detected by
//                    magic) and one or more "image" parts plus every /generate field
//                    (angles, blend, seed, resolution, texture, faces, atlas, ...), and
//                    "refine_coords" (auto|encoder|voxel) and "restore_transform" (on|off).
//                    Re-generates the mesh from the view(s) on the mesh's own voxels
//                    (docs/spec/31-mesh-refine.md). model=pixal3d -> 400.
//
// Launch-time defaults come from CLI flags (see trellis::parse_args);
// each request copies those defaults and applies its own overrides. The model
// directory is resolved once; each request runs the full pipeline via trellis_run()
// (per-stage load/free, like trellis-cli), serialized by a mutex. Keeping the
// process resident avoids re-initializing the Vulkan backend on every request.
#include "trellis_args.h"
#include "trellis_run.h"
#include "httplib.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>
#include <string>
#include <stdexcept>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

// std::tmpnam on MSVC yields drive-root paths ("\sXXX.N") that a non-elevated
// process cannot write; stage scratch files in the real temp directory instead.
std::string temp_stem(const std::string& workdir = {}) {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = workdir.empty() ? std::filesystem::temp_directory_path(ec)
                                                : std::filesystem::absolute(workdir);
    if (ec) dir = ".";
    std::filesystem::create_directories(dir,ec);
    if (ec) throw std::runtime_error("cannot create request workdir: " + ec.message());
    auto n = counter.fetch_add(1);
    return (dir / ("trellis-req-" + std::to_string(n))).string();
}

void bad_request(httplib::Response& res, const std::string& msg) {
    res.status = 400;
    res.set_content(msg + "\n", "text/plain");
}

// Field overrides shared by every endpoint that runs the pipeline. Returns false (with a 400
// already set on `res`) on an invalid value.
bool apply_common_fields(const httplib::Request& req, trellis::TrellisParams& p, httplib::Response& res) {
    if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
    if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
    if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;
    if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
    if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
    if (req.has_file("atlas")) {
        std::string error;
        if (!trellis::parse_atlas_arg(req.get_file_value("atlas").content.c_str(), p, error)) {
            bad_request(res, error);
            return false;
        }
    }
    if (req.has_file("faces")) {
        std::string error;
        if (!trellis::parse_faces_arg(req.get_file_value("faces").content.c_str(), p, error) ||
            !trellis::check_faces_compat(p, error)) {   // launch-time --decim + per-request faces
            bad_request(res, error);
            return false;
        }
    }
    // Both families share one model directory (the Pixal3D flows carry a pixal3d_ prefix),
    // so a single resident server can serve either -- the client just has to say which.
    if (req.has_file("model")) {
        const std::string& m = req.get_file_value("model").content;
        if      (m == "pixal3d") p.family = trellis::ModelFamily::Pixal3D;
        else if (m == "trellis") p.family = trellis::ModelFamily::Trellis;
        else { bad_request(res, "model must be trellis or pixal3d"); return false; }
    }
    // Camera, for pixal3d. Unlike every other knob these describe the IMAGE rather than the
    // run, so a launch-time default is close to useless on a server taking arbitrary uploads.
    for (const auto& field : {std::pair<const char*, const char*>{"fov", "fov"},
                              {"mesh_scale", "mesh-scale"}, {"extend_pixel", "extend-pixel"}}) {
        if (!req.has_file(field.first)) continue;
        std::string error;
        if (!trellis::parse_camera_arg(field.second, req.get_file_value(field.first).content.c_str(), p, error)) {
            bad_request(res, error);
            return false;
        }
    }
    if (req.has_file("texture")) {
        const std::string& t = req.get_file_value("texture").content;
        if      (t == "off" || t == "0" || t == "false") p.texture = false;
        else if (t == "on"  || t == "1" || t == "true")  p.texture = true;
        else { bad_request(res, "texture must be on or off"); return false; }
    }
    if (req.has_file("normal_map")) {
        const std::string& n = req.get_file_value("normal_map").content;
        if      (n == "off" || n == "0" || n == "false") p.normal_map = 0;
        else if (n == "on"  || n == "1" || n == "true")  p.normal_map = 1;
        else { bad_request(res, "normal_map must be on or off"); return false; }
    }
    if (req.has_file("webp")) {
        const std::string& w = req.get_file_value("webp").content;
        p.webp = (w == "off" || w == "0" || w == "false") ? 0
               : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
    }
    // Multi-view blend knobs (meaningful with several images).
    if (req.has_file("blend_temp")) {
        const std::string& v = req.get_file_value("blend_temp").content;
        char* end = nullptr;
        const float f = std::strtof(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(f) || f <= 0) { bad_request(res, "blend_temp must be a positive number"); return false; }
        p.blend_temp = f;
    }
    if (req.has_file("front_axis")) {
        const std::string& v = req.get_file_value("front_axis").content;
        trellis::FrontAxis fa;
        if (!trellis::parse_front_axis(v, fa)) { bad_request(res, "front_axis must be one of -y, +y, +x, -x, ref-z, ref-x"); return false; }
        p.front_axis = v;
    }
    if (req.has_file("yaw_offset")) {
        const std::string& v = req.get_file_value("yaw_offset").content;
        char* end = nullptr;
        const float f = std::strtof(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(f)) { bad_request(res, "yaw_offset must be a number of degrees"); return false; }
        p.yaw_offset = f;
    }
    return true;
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t b = 0;
    for (;;) {
        const size_t e = s.find(',', b);
        std::string t = s.substr(b, e == std::string::npos ? std::string::npos : e - b);
        while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
        while (!t.empty() && std::isspace((unsigned char)t.back())) t.pop_back();
        out.push_back(t);
        if (e == std::string::npos) break;
        b = e + 1;
    }
    return out;
}

// The angles of `n` uploaded images from the "azimuths" / "elevations" fields. Each azimuth goes
// through parse_view_spec (so names work and the CLI's rules apply). Returns false with `error`.
bool parse_view_angles(const httplib::Request& req, size_t n, std::vector<trellis::ViewSpec>& out,
                       std::string& error) {
    out.assign(n, trellis::ViewSpec{});
    const bool has_az = req.has_file("azimuths"), has_el = req.has_file("elevations");
    if (n > 1 && !has_az) { error = "azimuths is required when more than one image is sent"; return false; }
    if (has_az) {
        const auto az = split_commas(req.get_file_value("azimuths").content);
        if (az.size() != n) {
            error = "azimuths has " + std::to_string(az.size()) + " entries for " + std::to_string(n) + " image(s)";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            trellis::ViewSpec v;
            if (!trellis::parse_view_spec("x:" + az[i], v, error) || v.image != "x") {
                error = "bad azimuth '" + az[i] + "'";
                return false;
            }
            out[i].az_deg = v.az_deg;
            out[i].el_deg = v.el_deg;   // a name like "top" carries its elevation
        }
    }
    if (has_el) {
        const auto el = split_commas(req.get_file_value("elevations").content);
        if (el.size() != n) {
            error = "elevations has " + std::to_string(el.size()) + " entries for " + std::to_string(n) + " image(s)";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            char* end = nullptr;
            const float f = std::strtof(el[i].c_str(), &end);
            if (el[i].empty() || *end || !std::isfinite(f) || f < -90 || f > 90) {
                error = "bad elevation '" + el[i] + "' (degrees within [-90, 90])";
                return false;
            }
            out[i].el_deg = f;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // Stage progress goes to stdout, which is fully buffered when piped (e.g.
    // under Lemonade's output capture) — keep it line-visible for diagnostics.
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::TrellisParams base;
    if (!trellis::parse_args(argc, argv, base)) {
        trellis::print_usage(argv[0], /*server=*/true);
        return base.help ? 0 : 1;
    }

    std::mutex gen_mu;
    httplib::Server svr;

    // Trellis Studio (and any browser client) calls this server from a different
    // origin — a Tauri webview is tauri://localhost / http://tauri.localhost, and a
    // browser-served UI is another port — so every response needs permissive CORS
    // headers, and a multipart POST with non-simple headers may be preflighted with
    // OPTIONS. Applied to every route via the post-routing hook + a catch-all OPTIONS.
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "86400");
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;  // headers added by the post-routing handler above
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    // /generate and /refine share everything but the mesh part.
    auto run_request = [&](const httplib::Request& req, httplib::Response& res, bool refine) {
        if (refine && !req.has_file("mesh")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'mesh' file part (GLB or OBJ)\"}", "application/json");
            return;
        }
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'image' file part\"}", "application/json");
            return;
        }
        const auto images = req.get_file_values("image");

        // Per-request params start from the launch defaults, then apply overrides.
        trellis::TrellisParams p = base;
        if (!apply_common_fields(req, p, res)) return;
        std::vector<trellis::ViewSpec> angles;
        {
            std::string error;
            if (!parse_view_angles(req, images.size(), angles, error)) { bad_request(res, error); return; }
            if ((int)images.size() > trellis::kMaxViews) {
                bad_request(res, "at most " + std::to_string(trellis::kMaxViews) + " images are supported");
                return;
            }
            if ((images.size() > 1 || refine) && p.family == trellis::ModelFamily::Pixal3D) {
                bad_request(res, "--model pixal3d does not support multi-view conditioning (--view) or --refine-mesh yet;"
                                 " use --model trellis, or pass a single image.");
                return;
            }
        }

        std::string mesh_bytes;
        if (refine) {
            mesh_bytes = req.get_file_value("mesh").content;
            if (mesh_bytes.size() < 4) { bad_request(res, "mesh part is empty"); return; }
            if (req.has_file("refine_coords")) {
                const std::string& v = req.get_file_value("refine_coords").content;
                if      (v == "auto")    p.refine_coords = -1;
                else if (v == "voxel")   p.refine_coords = 0;
                else if (v == "encoder") p.refine_coords = 1;
                else { bad_request(res, "refine_coords must be auto, encoder or voxel"); return; }
            }
            if (req.has_file("restore_transform")) {
                const std::string& v = req.get_file_value("restore_transform").content;
                if      (v == "on" || v == "1" || v == "true")   p.refine_restore = true;
                else if (v == "off" || v == "0" || v == "false") p.refine_restore = false;
                else { bad_request(res, "restore_transform must be on or off"); return; }
            }
        }

        std::string stem;
        try {
            stem = temp_stem(p.retopo ? p.retopo_workdir : std::string{});
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to stage request files\"}", "application/json");
            return;
        }
        // One image and no angle fields: exactly the historical request (p.image, no views).
        // Otherwise every image is a view: <stem>_v<i>.png at its azimuth/elevation.
        const bool multi = images.size() > 1 || req.has_file("azimuths") || req.has_file("elevations");
        std::vector<std::string> staged;
        if (multi) {
            p.image.clear();
            p.views.clear();
            for (size_t i = 0; i < images.size(); ++i) {
                staged.push_back(stem + "_v" + std::to_string(i) + ".png");
                p.views.push_back(trellis::ViewSpec{ staged.back(), angles[i].az_deg, angles[i].el_deg });
            }
        } else {
            p.image = stem + ".png";
            staged.push_back(p.image);
        }
        p.output = stem + ".glb";
        if (refine) {
            // Detected by magic: binary glTF, else treated as OBJ text (load_mesh re-checks).
            p.refine_mesh = stem + (mesh_bytes.compare(0, 4, "glTF") == 0 ? "_mesh.glb" : "_mesh.obj");
            staged.push_back(p.refine_mesh);
        }

        std::string glb;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            for (size_t i = 0; i < staged.size(); ++i) {
                const std::string& bytes = (refine && i + 1 == staged.size()) ? mesh_bytes : images[i].content;
                if (!write_file_bytes(staged[i], bytes)) {
                    for (const auto& f : staged) std::remove(f.c_str());
                    res.status = 500;
                    res.set_content("{\"error\":\"failed to stage input image\"}", "application/json");
                    return;
                }
            }
            fprintf(stderr, "[trellis-server] %s: %zu image(s), %zu-byte first image, seed %u, res %s, bg %s, uv %s, faces %d, atlas %s, webp %s, texture %s\n",
                    refine ? "refine" : "generate", images.size(), images[0].content.size(), p.seed, p.cascade ? std::to_string(p.hr_res).c_str() : "512",
                    p.birefnet < 0 ? "auto" : (p.birefnet ? "birefnet" : "threshold"), p.xatlas ? "xatlas" : "box",
                    trellis::qem_face_target(p), p.tex > 0 ? std::to_string(p.tex).c_str() : "auto",
                    p.webp < 0 ? "auto" : (p.webp ? "on" : "off"), p.texture ? "on" : "off");
            try {
                int rc = trellis_run(p);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate failed: %s\n", e.what());
                error_message = e.what();
            }
            for (const auto& f : staged) std::remove(f.c_str());
            std::remove(p.output.c_str());
            // trellis_run also writes sibling debug artifacts; clean them up too.
            std::remove((stem + ".ply").c_str());
            std::remove((stem + "_base.png").c_str());
        }

        if (glb.empty()) {
            res.status = 500;
            std::string escaped;
            for (char c : error_message) {
                switch (c) {
                    case '"':  escaped += "\\\""; break;
                    case '\\': escaped += "\\\\"; break;
                    case '\n': escaped += "\\n";  break;
                    case '\r': escaped += "\\r";  break;
                    case '\t': escaped += "\\t";  break;
                    default:
                        if ((unsigned char)c < 0x20) break;
                        escaped += c;
                }
            }
            res.set_content("{\"error\":\"" + escaped + "\"}", "application/json");
            return;
        }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    };
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) { run_request(req, res, false); });
    svr.Post("/refine",   [&](const httplib::Request& req, httplib::Response& res) { run_request(req, res, true); });

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            base.models.c_str(), base.gpu, base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
