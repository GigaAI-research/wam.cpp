#include "policy/action_decoder.h"

#include "wam/error.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace wam::internal::policy {
namespace {

bool active(const NormalizationStats & stats, std::size_t index) {
    return stats.mask.empty() || stats.mask[index] != 0;
}

} // namespace

void validate_core_action(const std::vector<float> & normalized_action,
                          const ActionSpec & spec) {
    if (spec.horizon != 0 &&
        spec.model_dim <= std::numeric_limits<std::size_t>::max() /
                              spec.horizon &&
        normalized_action.size() == spec.horizon * spec.model_dim) {
        for (std::size_t index = 0; index < normalized_action.size(); ++index) {
            if (!std::isfinite(normalized_action[index])) {
                throw Error(ErrorCode::inference_failed,
                            "model action contains NaN or Inf",
                            {{"action", std::to_string(index)}});
            }
        }
        return;
    }
    throw Error(ErrorCode::inference_failed,
                "model action shape does not match PolicySpec",
                {{"action", "expected horizon multiplied by model_dim"}});
}

DecodedActionChunk decode_action_reference(
    const std::vector<float> & normalized_action,
    const std::vector<float> & raw_state, const ActionSpec & spec,
    const NormalizationStats & stats) {
    validate_core_action(normalized_action, spec);
    if (spec.recovery.kind == ActionRecoveryKind::add_current_state &&
        spec.recovery.reference_state_indices.size() != spec.real_dim) {
        throw Error(ErrorCode::invalid_argument,
                    "action recovery mapping is invalid");
    }

    DecodedActionChunk result;
    result.horizon = spec.horizon;
    result.action_dim = spec.real_dim;
    result.values.resize(spec.horizon * spec.real_dim);
    for (std::size_t step = 0; step < spec.horizon; ++step) {
        for (std::size_t dimension = 0; dimension < spec.real_dim;
             ++dimension) {
            float value = normalized_action[step * spec.model_dim + dimension];
            if (active(stats, dimension)) {
                if (spec.normalization.output_clamp_lower.has_value()) {
                    value = std::max(
                        *spec.normalization.output_clamp_lower,
                        std::min(*spec.normalization.output_clamp_upper, value));
                }
                if (spec.normalization.clip &&
                    (spec.normalization.kind == NormalizationKind::min_max ||
                     spec.normalization.kind == NormalizationKind::quantile)) {
                    value = std::max(-1.0F, std::min(1.0F, value));
                }
                switch (spec.normalization.kind) {
                    case NormalizationKind::none:
                        break;
                    case NormalizationKind::z_score:
                        value = value * stats.stddev[dimension] +
                            stats.mean[dimension];
                        break;
                    case NormalizationKind::min_max:
                    case NormalizationKind::quantile:
                        value = ((value + 1.0F) * 0.5F) *
                                    (stats.upper[dimension] -
                                     stats.lower[dimension]) +
                                stats.lower[dimension];
                        break;
                }
            }
            if (spec.recovery.kind ==
                ActionRecoveryKind::add_current_state) {
                const std::int32_t state_index =
                    spec.recovery.reference_state_indices[dimension];
                if (state_index >= 0) {
                    if (static_cast<std::size_t>(state_index) >=
                        raw_state.size()) {
                        throw Error(
                            ErrorCode::invalid_argument,
                            "action recovery state index is out of range",
                            {{"action.recovery", std::to_string(state_index)}});
                    }
                    value += raw_state[static_cast<std::size_t>(state_index)];
                }
            }
            if (!std::isfinite(value)) {
                throw Error(ErrorCode::inference_failed,
                            "decoded action contains NaN or Inf",
                            {{"action", std::to_string(
                                step * spec.real_dim + dimension)}});
            }
            result.values[step * spec.real_dim + dimension] = value;
        }
    }
    return result;
}

Tensor make_action_tensor(const DecodedActionChunk & action) {
    if (action.horizon == 0 || action.action_dim == 0 ||
        action.action_dim > std::numeric_limits<std::size_t>::max() /
                                action.horizon ||
        action.values.size() != action.horizon * action.action_dim) {
        throw Error(ErrorCode::invalid_argument,
                    "policy action chunk shape is invalid");
    }
    Tensor tensor;
    tensor.dtype = DType::f32;
    tensor.shape = {static_cast<std::int64_t>(action.horizon),
                    static_cast<std::int64_t>(action.action_dim)};
    tensor.layout = "T,A";
    tensor.byte_order = ByteOrder::little;
    tensor.data.resize(action.values.size() * sizeof(float));
    std::memcpy(tensor.data.data(), action.values.data(), tensor.data.size());
    return tensor;
}

} // namespace wam::internal::policy
