#pragma once

#include "policy/policy_spec.h"
#include "wam/observation.h"

#include <vector>

namespace wam::internal::policy {

void validate_state_input(const TensorView & state, const StateSpec & spec);
std::vector<float> read_state_f32(const TensorView & state);
std::vector<float> pad_state(const std::vector<float> & state,
                             const StateSpec & spec);
void normalize_state_reference(std::vector<float> & state,
                               const StateSpec & spec,
                               const NormalizationStats & stats);

} // namespace wam::internal::policy
