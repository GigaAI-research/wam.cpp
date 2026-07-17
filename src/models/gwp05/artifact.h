#pragma once

#include "semantics.h"

#include "wam/types.h"

#include <memory>
#include <string>
#include <vector>

namespace wam::internal {

class GgufReader;

} // namespace wam::internal

namespace wam::internal::gwp05 {

struct Geometry {
    std::uint32_t hidden = 0;
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t ffn_dim = 0;
    std::uint32_t action_hidden = 0;
    std::uint32_t action_ffn_dim = 0;
    std::uint32_t action_dim = 0;
    std::uint32_t real_state_dim = 0;
    std::uint32_t real_action_dim = 0;
    std::uint32_t num_embodiments = 0;
    std::uint32_t embodiment_id = 0;
    std::uint32_t image_height = 0;
    std::uint32_t image_width = 0;
    std::uint32_t num_views = 0;
    std::uint32_t action_chunk = 0;
    std::uint32_t inference_steps = 0;
    float flow_shift = 0.0f;
    float norm_eps = 0.0f;
    std::uint32_t t5_vocab_size = 0;
    std::uint32_t t5_hidden = 0;
    std::uint32_t t5_ffn_dim = 0;
    std::uint32_t t5_heads = 0;
    std::uint32_t t5_head_dim = 0;
    std::uint32_t t5_layers = 0;
    std::uint32_t t5_max_length = 0;
    std::uint32_t vae_z_dim = 0;
};

struct ArtifactContract {
    std::shared_ptr<GgufReader> reader;
    Geometry geometry;
    semantics::SequenceGeometry sequence_geometry;
    std::string policy;
    std::vector<ArtifactComponentInfo> components;
};

ArtifactContract inspect_artifact(std::shared_ptr<GgufReader> reader,
                                  const ModelOptions & options);

} // namespace wam::internal::gwp05
