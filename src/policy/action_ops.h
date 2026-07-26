#pragma once

#include "policy/policy_spec.h"

#include <cstddef>
#include <random>
#include <vector>

namespace wam::internal::policy {

struct PolicyActionChunk {
    std::vector<float> values;
    std::size_t horizon = 0;
    std::size_t action_dim = 0;
};

void validate_core_action(const std::vector<float> & normalized_action,
                          const ActionSpec & spec);
PolicyActionChunk decode_action_reference(
    const std::vector<float> & normalized_action,
    const std::vector<float> & raw_state,
    const ActionSpec & spec,
    const NormalizationStats & stats);
Tensor make_action_tensor(const PolicyActionChunk & action);

std::vector<float> prepare_action_noise(const TensorView & explicit_noise,
                                        const ActionSpec & spec,
                                        std::mt19937 & session_rng);

} // namespace wam::internal::policy
