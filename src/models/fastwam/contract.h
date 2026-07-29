#pragma once

#include "models/fastwam/semantics.h"
#include "policy/policy_spec.h"
#include "wam/model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam::internal {

class GgufReader;

namespace fastwam {

struct Geometry {
    std::uint32_t image_height = 0;
    std::uint32_t image_width = 0;
    std::uint32_t num_cameras = 0;
    std::uint32_t action_dim = 0;
    std::uint32_t proprio_dim = 0;
    std::uint32_t action_horizon = 0;
    std::uint32_t inference_steps = 0;
    std::uint32_t latent_channels = 0;
    std::uint32_t spatial_downsample = 0;
    std::uint32_t temporal_downsample = 0;
    std::uint32_t context_len = 0;
    std::uint32_t text_dim = 0;
    std::uint32_t video_hidden_dim = 0;
    std::uint32_t action_hidden_dim = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t attn_head_dim = 0;
    float action_shift = 0.0F;
    float video_shift = 0.0F;
    float norm_eps = 0.0F;
    std::string variant;
};

struct FastWamContract {
    std::shared_ptr<GgufReader> reader;
    Geometry geometry;
    semantics::SequenceGeometry sequence_geometry;
    std::vector<float> proprio_weight;
    std::vector<float> proprio_bias;
    std::string conversion_policy;
    std::vector<ArtifactComponentInfo> components;
};

std::shared_ptr<const FastWamContract> load_contract(
    std::shared_ptr<GgufReader> reader,
    const policy::PolicySpec & policy_spec);
void validate_contract(const FastWamContract & artifact,
                       const policy::PolicySpec & policy_spec);

} // namespace fastwam
} // namespace wam::internal
