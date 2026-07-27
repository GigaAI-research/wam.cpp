#pragma once

#include "wam/types.h"

#include <cstdint>
#include <vector>

namespace wam::internal::model_common {

struct CoreAction {
    std::vector<float> values;
    Stats stats;
};

struct EngineInfo {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::unknown;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;
    std::vector<RuntimeComponentInfo> runtime_components;
};

} // namespace wam::internal::model_common
