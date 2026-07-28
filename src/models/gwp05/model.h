#pragma once

#include "model_internal.h"

#include <memory>
#include <optional>

namespace wam::internal {

class GgufReader;

namespace gwp05 {

std::unique_ptr<ModelImpl> create_model(
    const RuntimeConfig & options,
    ModelInfo info,
    std::optional<policy::PolicySpec> policy_spec,
    std::shared_ptr<GgufReader> reader);

} // namespace gwp05

} // namespace wam::internal
