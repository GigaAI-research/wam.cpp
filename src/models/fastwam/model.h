#pragma once

#include "model_internal.h"

#include <memory>
#include <optional>

namespace wam::internal {

class GgufReader;
class ModelRegistry;

namespace fastwam {

std::unique_ptr<ModelImpl> create_model(
    const ModelOptions & options,
    ModelInfo info,
    std::optional<policy::PolicySpecDraft> policy_spec,
    std::shared_ptr<GgufReader> reader);

} // namespace fastwam

void register_fastwam(ModelRegistry & registry);

} // namespace wam::internal
