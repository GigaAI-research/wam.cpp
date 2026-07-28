#pragma once

#include "policy/policy_spec.h"
#include "wam/prediction.h"

#include <cstddef>
#include <vector>

namespace wam::internal::policy {

struct DecodedActionChunk {
    std::vector<float> values;
    std::size_t horizon = 0;
    std::size_t action_dim = 0;
};

void validate_core_action(const std::vector<float> & normalized_action,
                          const ActionSpec & spec);
DecodedActionChunk decode_action_reference(
    const std::vector<float> & normalized_action,
    const std::vector<float> & raw_state,
    const ActionSpec & spec,
    const NormalizationStats & stats);
Tensor make_action_tensor(const DecodedActionChunk & action);

} // namespace wam::internal::policy
