#pragma once

#include "models/gwp05/semantics.h"
#include "policy/policy_spec.h"

#include <cstdint>
#include <memory>
#include <string>

namespace wam::internal {

class GgufReader;

namespace gwp05 {

struct Geometry {
    std::uint32_t hidden = 0;
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t ffn_dim = 0;
    std::uint32_t action_hidden = 0;
    std::uint32_t action_ffn_dim = 0;
    std::uint32_t state_dim = 0;
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
    float flow_shift = 0.0F;
    float norm_epsilon = 0.0F;
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
    std::string conversion_policy;
    bool legacy_policy_spec = false;
};

policy::PolicySpec read_legacy_policy_spec(const GgufReader & reader);
std::shared_ptr<const ArtifactContract> load_artifact(
    std::shared_ptr<GgufReader> reader,
    const policy::PolicySpec & policy_spec);
void validate_artifact(const ArtifactContract & artifact,
                       const policy::PolicySpec & policy_spec);

} // namespace gwp05
} // namespace wam::internal
