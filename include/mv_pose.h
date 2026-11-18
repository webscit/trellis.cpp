// Multi-view pose checks from silhouettes (docs/spec/30-multiview.md, "Pose checks").
//
// TRELLIS.2 decides the object's pose itself: the sparse-structure flow puts the object in its
// own canonical orientation whatever the views say. The views' azimuths only steer the blend, so
// they must be expressed in that frame. These helpers compare the occupancy of a front-only
// sparse-structure pass with each view's matte:
//
//   * project the occupied voxels orthographically along a candidate view direction,
//   * crop both silhouettes to their bounding box and rescale them to a common square (aspect
//     kept), so scale and framing drop out,
//   * score the pair by IoU.
//
// From that table: a global yaw offset (`--yaw-offset auto`), per-view azimuths (`--view
// img:auto`) and a sanity check of the supplied ones.
//
// Limits of silhouettes. An orthographic silhouette from az+180 is the mirror image of the one
// from az, and for an object that is left/right symmetric the silhouette from 180-az equals the
// one from az. So front and back cannot be told apart on symmetric objects: such ties are
// resolved toward the supplied value (yaw 0, or the user's azimuth) and reported as ambiguous.
#pragma once
#include "multiview.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

constexpr int kSilSize = 96;   // side of the normalised silhouettes

// A bbox-normalised binary silhouette, kSilSize x kSilSize, row-major (row 0 = image top).
using Silhouette = std::vector<uint8_t>;

// From an RGBA image (alpha > 127 = object).
Silhouette matte_silhouette(const unsigned char* rgba, int w, int h);
// Orthographic projection of occupied cells of a G^3 grid (cell (x,y,z), grid axes as in
// view_dir_grid) seen from view (az, el); bp.front and bp.yaw_offset apply. Image right = the
// camera's right, image up = world up (for el = +-90 the in-plane rotation is a convention).
Silhouette voxel_silhouette(const std::vector<std::array<int,3>>& cells, int G,
                            float az_deg, float el_deg, const BlendParams& bp);
// IoU of two silhouettes; 0 when both are empty.
float silhouette_iou(const Silhouette& a, const Silhouette& b);

// Occupied cells of the sparse-structure decoder's 64^3 logits (logit > 0), token order
// x*64*64 + y*64 + z as in ss_coords.
std::vector<std::array<int,3>> occupied_cells(const std::vector<float>& logits, int S);

// IoU of view v's matte against the occupancy seen from every candidate azimuth
// az = k * step (k = 0 .. 360/step - 1) at the view's elevation, with yaw offset 0.
struct PoseTable {
    int step = 15;
    std::vector<std::vector<float>> iou;   // [view][k]
    float at(int v, float az) const;       // nearest candidate, az wrapped to [0, 360)
};
PoseTable pose_table(const std::vector<std::array<int,3>>& cells, int G,
                     const std::vector<ViewSpec>& views, const std::vector<Silhouette>& mattes,
                     FrontAxis front, int step = 15);

// Global yaw search over {0, 90, 180, 270}: the mean IoU over views whose azimuth is known
// (`known[v]`) at az_v + yaw. A candidate replaces 0 only when it beats it by `margin`.
struct YawChoice { float yaw = 0; float score0 = 0, best = 0; std::vector<float> scores; bool ambiguous = false; };
YawChoice choose_yaw(const PoseTable& t, const std::vector<ViewSpec>& views,
                     const std::vector<bool>& known, float margin = 0.03f);

// Best azimuth for one view (with the yaw already applied to the table lookups). Ties within
// `margin` of the best are collected in `alternatives`; among them the one farthest from the
// other views' azimuths is chosen (views are normally taken from different sides).
struct AzChoice { float az = 0; float iou = 0; std::vector<float> alternatives; };
AzChoice choose_azimuth(const PoseTable& t, int v, float yaw, const std::vector<float>& other_az,
                        float margin = 0.02f);

// ---- front-outline fill (--mv-ss fill) ----
// Cells of `front` to add to `blended` so the blended occupancy covers the front pass's
// silhouette as seen from view (az, el): every line of sight (cell-sized pixel) that hits the
// front occupancy but misses the blended one gets the front pass's cells on that line.
// Returns indices into `front`.
std::vector<size_t> front_fill(const std::vector<std::array<int,3>>& blended,
                               const std::vector<std::array<int,3>>& front, int G,
                               float az_deg, float el_deg, const BlendParams& bp);

// Plain-text dump of a silhouette pair (debugging aid; '#' both, 'm' matte only, 'v' voxels only).
std::string silhouette_diff_ascii(const Silhouette& matte, const Silhouette& vox, int every = 3);

}  // namespace trellis
