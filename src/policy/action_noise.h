#pragma once

#include "policy/policy_spec.h"
#include "wam/observation.h"

#include <random>
#include <vector>

namespace wam::internal::policy {

std::vector<float> prepare_action_noise(const TensorView & explicit_noise,
                                        const ActionSpec & spec,
                                        std::mt19937 & session_rng);

} // namespace wam::internal::policy
