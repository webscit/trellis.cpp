// 3MF writer (core spec + materials extension) for print-ready meshes: an OPC/ZIP package
// holding the model XML and, when textured, the base-colour PNG. See mesh_print.h.
#include "mesh_print.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Defined by stb_image_write's implementation unit (mesh_glb.cpp).
extern "C" unsigned char* stbi_zlib_compress(unsigned char* data, int data_len, int* out_len, int quality);
extern "C" unsigned char* stbi_write_png_to_mem(const unsigned char* pixels, int stride_bytes, int x, int y, int n,
                                     int* out_len, const char* parameters);

namespace trellis {

namespace {

uint32_t crc32_of(const unsigned char* data, size_t size) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        ready = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// Minimal ZIP writer (deflate via stb's zlib encoder, stored when that does not help).
class ZipWriter {
public:
    explicit ZipWriter(const char* path) : file_(std::fopen(path, "wb")) {}
    ~ZipWriter() { if (file_) std::fclose(file_); }
    bool open() const { return file_ != nullptr; }

    void add(const std::string& name, const unsigned char* data, size_t size, bool compress) {
        Entry e;
        e.name = name;
        e.crc = crc32_of(data, size);
        e.size = (uint32_t)size;
        e.offset = (uint32_t)offset_;
        const unsigned char* payload = data;
        size_t stored = size;
        unsigned char* packed = nullptr;
        if (compress && size > 0 && size < 0x7FFFFFFFu) {
            int length = 0;
            packed = stbi_zlib_compress(const_cast<unsigned char*>(data), (int)size, &length, 8);
            // zlib stream = 2-byte header + raw deflate + 4-byte adler32
            if (packed && length > 6 && (size_t)(length - 6) < size) {
                payload = packed + 2;
                stored = (size_t)length - 6;
                e.method = 8;
            }
        }
        e.packed = (uint32_t)stored;
        put32(0x04034b50u);
        header(e);
        write(e.name.data(), e.name.size());
        write(payload, stored);
        std::free(packed);
        entries_.push_back(e);
    }

    bool close() {
        if (!file_) return false;
        const size_t start = offset_;
        for (const Entry& e : entries_) {
            put32(0x02014b50u); put16(20);
            header(e);
            put16(0); put16(0); put16(0); put32(0); put32(e.offset);
            write(e.name.data(), e.name.size());
        }
        const size_t size = offset_ - start;
        put32(0x06054b50u); put16(0); put16(0);
        put16((uint16_t)entries_.size()); put16((uint16_t)entries_.size());
        put32((uint32_t)size); put32((uint32_t)start); put16(0);
        const bool good = std::fclose(file_) == 0 && ok_;
        file_ = nullptr;
        return good;
    }

private:
    struct Entry { std::string name; uint32_t crc = 0, size = 0, packed = 0, offset = 0; uint16_t method = 0; };
    // The fields the local and the central header share: version .. extra length.
    void header(const Entry& e) {
        put16(20); put16(0); put16(e.method); put16(0); put16(0x21);   // time 0, date 1980-01-01
        put32(e.crc); put32(e.packed); put32(e.size);
        put16((uint16_t)e.name.size()); put16(0);
    }
    void write(const void* data, size_t size) {
        if (size && std::fwrite(data, 1, size, file_) != size) ok_ = false;
        offset_ += size;
    }
    void put16(uint16_t v) { const unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)}; write(b, 2); }
    void put32(uint32_t v) {
        const unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
        write(b, 4);
    }
    FILE* file_;
    size_t offset_ = 0;
    bool ok_ = true;
    std::vector<Entry> entries_;
};

const char kTypes[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n"
    " <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n"
    " <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n"
    " <Default Extension=\"png\" ContentType=\"image/png\"/>\n"
    "</Types>\n";
const char kRels[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n"
    " <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\""
    " Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n"
    "</Relationships>\n";
const char kModelRels[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n"
    " <Relationship Target=\"/3D/Textures/base_color.png\" Id=\"rel1\""
    " Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture\"/>\n"
    "</Relationships>\n";

}  // namespace

bool write_3mf(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               float size_mm, const PrintTexture* texture) {
    if (V <= 0 || F <= 0) return false;
    const bool textured = texture && texture->ok();

    // Placement: z-up, on the plate, centred, longest side = size_mm.
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int64_t v = 0; v < V; ++v)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], verts[3 * v + k]);
            hi[k] = std::max(hi[k], verts[3 * v + k]);
        }
    const float extent = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    const float scale = size_mm > 0 && extent > 0 ? size_mm / extent : 1.0f;
    const float origin[3] = {0.5f * (lo[0] + hi[0]), 0.5f * (lo[1] + hi[1]), lo[2]};

    // 3MF meshes share vertices by index: merge the UV-seam duplicates so the solid stays
    // closed. The UVs keep the caller's (split) indexing, one tex2coord per input vertex.
    std::vector<float> welded(verts, verts + 3 * V);
    std::vector<int32_t> tris(faces, faces + 3 * F);
    weld_exact(welded, tris);

    std::string model;
    model.reserve((size_t)F * 72 + (size_t)V * 96);
    char line[256];
    model += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<model unit=\"millimeter\" xml:lang=\"en-US\""
             " xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\""
             " xmlns:m=\"http://schemas.microsoft.com/3dmanufacturing/material/2015/02\">\n"
             " <metadata name=\"Application\">trellis.cpp</metadata>\n <resources>\n";
    if (textured) {
        model += "  <m:texture2d id=\"1\" path=\"/3D/Textures/base_color.png\" contenttype=\"image/png\"/>\n"
                 "  <m:texture2dgroup id=\"2\" texid=\"1\">\n";
        for (int64_t v = 0; v < V; ++v) {   // 3MF's v runs up from the bottom-left corner
            std::snprintf(line, sizeof line, "   <m:tex2coord u=\"%.6f\" v=\"%.6f\"/>\n",
                          texture->uv[2 * v], 1.0f - texture->uv[2 * v + 1]);
            model += line;
        }
        model += "  </m:texture2dgroup>\n";
    }
    model += "  <object id=\"3\" type=\"model\">\n   <mesh>\n    <vertices>\n";
    for (size_t v = 0; v < welded.size() / 3; ++v) {
        std::snprintf(line, sizeof line, "     <vertex x=\"%.5f\" y=\"%.5f\" z=\"%.5f\"/>\n",
                      (welded[3 * v] - origin[0]) * scale, (welded[3 * v + 1] - origin[1]) * scale,
                      (welded[3 * v + 2] - origin[2]) * scale);
        model += line;
    }
    model += "    </vertices>\n    <triangles>\n";
    for (int64_t f = 0; f < F; ++f) {
        const int32_t* t = &tris[3 * (size_t)f];
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        if (textured)
            std::snprintf(line, sizeof line,
                          "     <triangle v1=\"%d\" v2=\"%d\" v3=\"%d\" pid=\"2\" p1=\"%d\" p2=\"%d\" p3=\"%d\"/>\n",
                          t[0], t[1], t[2], faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]);
        else
            std::snprintf(line, sizeof line, "     <triangle v1=\"%d\" v2=\"%d\" v3=\"%d\"/>\n", t[0], t[1], t[2]);
        model += line;
    }
    model += "    </triangles>\n   </mesh>\n  </object>\n </resources>\n"
             " <build>\n  <item objectid=\"3\"/>\n </build>\n</model>\n";

    ZipWriter zip(path);
    if (!zip.open()) return false;
    zip.add("[Content_Types].xml", (const unsigned char*)kTypes, sizeof kTypes - 1, true);
    zip.add("_rels/.rels", (const unsigned char*)kRels, sizeof kRels - 1, true);
    zip.add("3D/3dmodel.model", (const unsigned char*)model.data(), model.size(), true);
    if (textured) {
        int length = 0;
        unsigned char* png = stbi_write_png_to_mem(texture->rgba, texture->size * 4, texture->size,
                                                   texture->size, 4, &length, nullptr);
        if (!png) return false;
        zip.add("3D/_rels/3dmodel.model.rels", (const unsigned char*)kModelRels, sizeof kModelRels - 1, true);
        zip.add("3D/Textures/base_color.png", png, (size_t)length, false);   // already deflated
        std::free(png);
    }
    return zip.close();
}

}  // namespace trellis
