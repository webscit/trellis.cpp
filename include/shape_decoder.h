// FlexiDualGrid shape VAE decoder: shape SLAT latent -> dual-grid head [7, M] @ res 512.
#pragma once
#include <vector>
#include <array>
#include <cstdint>

namespace trellis {
struct Model;

struct ShapeOut {
    std::vector<float> feats7;                 // [7*M] channel-major (ne0=7): per voxel dual-grid fields
    std::vector<std::array<int,3>> coords;     // [M] voxel coords at final resolution
    int res = 512;
    std::vector<std::vector<uint8_t>> subs;    // per-C2S binarized subdiv masks (for the tex decoder)
};

// latent: shape SLAT feats [32*N0] channel-major; coords0: active voxels. resolution = final
// grid size (coords0 res * 16): 512 for res-32 input, 1024 for res-64 input (cascade HR).
ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution = 512);

// Cascade upsample: run from_latent + the 4 C2S up-blocks and return the grown voxel coords
// (input res * 16), discarding feats. Used by the 1024 cascade to get fine coords from the LR slat.
std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0);

// Texture (PBR) decoder: SparseUnetVaeDecoder driven by the shape decoder's `subs` so it grows
// the IDENTICAL voxel tree. tex_latent: [32*N0]; returns 6-ch PBR [6*M] (channel-major) at the
// SAME final coords/order as shape_decode (base_color3, metallic, roughness, alpha) — pre *0.5+0.5.
std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs);

// The decoder's voxel tree when there is no shape decode to replay: texture-only mode
// (docs/spec/33-texture-only.md). Upstream's tex decoder grows exactly the fine voxels that the
// shape ENCODER merged (its S2C spatial cache), so the tree is the fine set halved level by
// level. `subs[k]` is the [8*N_k] child mask of decoder stage k in sparse_c2s's layout
// (o + 8*i, child (2x + (o&1), 2y + (o>>1&1), 2z + (o>>2&1))), walked from `latent` in its
// given order; `coords` is the order of tex_decode's output rows (a permutation of `fine`).
struct DecodeTree {
    std::vector<std::vector<uint8_t>> subs;     // 4 stages
    std::vector<std::array<int,3>> coords;      // final voxels, tex_decode row order
};
// Throws if `latent` (the encoder's coords) is not exactly the set of `fine` >> 4.
DecodeTree decode_tree_from_fine(const std::vector<std::array<int,3>>& fine,
                                 const std::vector<std::array<int,3>>& latent);

} // namespace trellis
