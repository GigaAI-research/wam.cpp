#pragma once

#include "models/fastwam/artifact.h"
#include "models/common/model_types.h"
#include "policy/observation_processor.h"
#include "wam/observation.h"
#include "wam/prediction.h"
#include "wam/runtime_config.h"

#include <random>
#include <vector>

namespace wam::internal::fastwam {

struct PreparedInputs {
    policy::PreparedObservation observation;
    Tensor embedding;
    std::vector<std::int32_t> embedding_attention_mask;
};

using CoreAction = model_common::CoreAction;

PreparedInputs prepare_inputs(const Observation & inputs,
                              const ArtifactContract & artifact,
                              const policy::PolicySpec & policy_spec,
                              LanguageRuntimeMode language_mode,
                              std::mt19937 & session_rng);

} // namespace wam::internal::fastwam
