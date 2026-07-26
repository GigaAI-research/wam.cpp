#pragma once

#include "models/fastwam/artifact.h"
#include "policy/image_ops.h"
#include "wam/types.h"

#include <random>
#include <vector>

namespace wam::internal::fastwam {

struct PreparedInputs {
    policy::CpuImage composite_image;
    std::vector<float> raw_state;
    std::vector<float> model_state;
    Tensor embedding;
    std::vector<std::int32_t> embedding_attention_mask;
    std::vector<float> action_noise;
};

struct CoreAction {
    std::vector<float> values;
    Stats stats;
};

PreparedInputs prepare_inputs(const Inputs & inputs,
                              const ArtifactContract & artifact,
                              const policy::PolicySpecDraft & policy_spec,
                              LanguageRuntimeMode language_mode,
                              std::mt19937 & session_rng);

} // namespace wam::internal::fastwam
