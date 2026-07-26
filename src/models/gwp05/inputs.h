#pragma once

#include "models/gwp05/artifact.h"
#include "policy/action_ops.h"
#include "policy/image_ops.h"
#include "wam/types.h"

#include <optional>
#include <random>
#include <vector>

namespace wam::internal::gwp05 {

struct PreparedInputs {
    policy::CpuImage composite_image;
    std::vector<float> raw_state;
    std::vector<float> model_state;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
    Tensor embedding;
    std::vector<std::int32_t> embedding_attention_mask;
    std::vector<float> action_noise;
    bool enable_prefix_cache = true;
};

struct CoreAction {
    std::vector<float> values;
    std::uint32_t horizon = 0;
    std::uint32_t model_action_dim = 0;
    Stats stats;
};

PreparedInputs prepare_inputs(const Inputs & inputs,
                              const ArtifactContract & artifact,
                              const policy::PolicySpecDraft & policy_spec,
                              LanguageRuntimeMode language_mode,
                              std::mt19937 & session_rng,
                              const std::optional<FixedPrompt> & fixed_prompt =
                                  std::nullopt);

} // namespace wam::internal::gwp05
