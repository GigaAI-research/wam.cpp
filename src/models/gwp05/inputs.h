#pragma once

#include "models/gwp05/artifact.h"
#include "models/common/model_types.h"
#include "policy/observation_processor.h"
#include "wam/observation.h"
#include "wam/prediction.h"
#include "wam/runtime_config.h"

#include <optional>
#include <random>
#include <vector>

namespace wam::internal::gwp05 {

struct PreparedInputs {
    policy::PreparedObservation observation;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
    Tensor embedding;
    std::vector<std::int32_t> embedding_attention_mask;
    bool enable_prefix_cache = true;
};

using CoreAction = model_common::CoreAction;

PreparedInputs prepare_inputs(const Observation & inputs,
                              const ArtifactContract & artifact,
                              const policy::PolicySpec & policy_spec,
                              LanguageRuntimeMode language_mode,
                              std::mt19937 & session_rng,
                              const std::optional<FixedPrompt> & fixed_prompt =
                                  std::nullopt);

} // namespace wam::internal::gwp05
