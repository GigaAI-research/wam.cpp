#pragma once

#include "policy/image_ops.h"
#include "policy/policy_spec.h"
#include "wam/observation.h"

#include <random>
#include <vector>

namespace wam::internal::policy {

struct PreparedObservation {
    CpuImage composite_image;
    std::vector<float> raw_state;
    std::vector<float> model_state;
    std::vector<float> action_noise;
};

PreparedObservation prepare_observation_reference(
    const Observation & inputs, const PolicySpec & policy_spec,
    std::mt19937 & session_rng);

} // namespace wam::internal::policy
