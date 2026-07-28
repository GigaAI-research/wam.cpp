#pragma once

#include "policy/policy_spec.h"

#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

struct ArtifactContract;

namespace semantics {

struct GeometryInput {
    std::uint32_t image_height = 0;
    std::uint32_t image_width = 0;
    std::uint32_t spatial_downsample = 0;
    std::uint32_t action_horizon = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t video_hidden_dim = 0;
};

struct SequenceGeometry {
    std::uint32_t latent_height = 0;
    std::uint32_t latent_width = 0;
    std::uint32_t video_grid_height = 0;
    std::uint32_t video_grid_width = 0;
    std::uint32_t video_tokens = 0;
    std::uint32_t action_tokens = 0;
};

struct VisualPositions {
    std::vector<std::int32_t> time;
    std::vector<std::int32_t> height;
    std::vector<std::int32_t> width;
};

SequenceGeometry resolve_geometry(const GeometryInput & input);
VisualPositions visual_positions(const SequenceGeometry & geometry);
std::vector<std::int32_t> action_positions(const SequenceGeometry & geometry);
void validate_policy_semantics(const policy::PolicySpec & policy_spec,
                               const ArtifactContract & artifact);

} // namespace semantics
} // namespace wam::internal::fastwam
