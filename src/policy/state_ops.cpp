#include "policy/state_ops.h"

#include "models/common/input_validation.h"
#include "wam/error.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace wam::internal::policy {
namespace {

bool active(const NormalizationStats & stats, std::size_t index) {
    return stats.mask.empty() || stats.mask[index] != 0;
}

} // namespace

void validate_state_input(const TensorView & state, const StateSpec & spec) {
    if (!state.layout.empty() && state.layout != "D") {
        throw Error(ErrorCode::invalid_argument,
                    "state tensor layout is invalid",
                    {{"state.layout", "expected D or empty"}});
    }
    (void) copy_f32_tensor(
        state, {{static_cast<std::int64_t>(spec.real_dim)}}, "state");
}

std::vector<float> read_state_f32(const TensorView & state) {
    if (state.shape.size() != 1) {
        throw Error(ErrorCode::invalid_argument,
                    "state tensor rank is invalid",
                    {{"state", "expected rank one"}});
    }
    return copy_f32_tensor(state, {state.shape}, "state");
}

std::vector<float> pad_state(const std::vector<float> & state,
                             const StateSpec & spec) {
    if (state.size() != spec.real_dim || spec.model_dim < spec.real_dim ||
        !std::isfinite(spec.pad_value)) {
        throw Error(ErrorCode::invalid_argument,
                    "state padding contract is invalid",
                    {{"state", "dimension or pad value mismatch"}});
    }
    std::vector<float> result(spec.model_dim, spec.pad_value);
    std::copy(state.begin(), state.end(), result.begin());
    return result;
}

void normalize_state_reference(std::vector<float> & state,
                               const StateSpec & spec,
                               const NormalizationStats & stats) {
    if (state.size() != spec.model_dim) {
        throw Error(ErrorCode::invalid_argument,
                    "model state dimension is invalid",
                    {{"state", "expected " + std::to_string(spec.model_dim)}});
    }
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (!active(stats, index)) continue;
        float value = state[index];
        switch (spec.normalization.kind) {
            case NormalizationKind::none:
                break;
            case NormalizationKind::z_score:
                value = (value - stats.mean[index]) / stats.stddev[index];
                break;
            case NormalizationKind::min_max:
            case NormalizationKind::quantile:
                value = ((value - stats.lower[index]) /
                         (stats.upper[index] - stats.lower[index])) * 2.0F - 1.0F;
                if (spec.normalization.clip) {
                    value = std::max(-1.0F, std::min(1.0F, value));
                }
                break;
        }
        if (spec.normalization.output_clamp_lower.has_value()) {
            value = std::max(*spec.normalization.output_clamp_lower,
                             std::min(*spec.normalization.output_clamp_upper,
                                      value));
        }
        if (!std::isfinite(value)) {
            throw Error(ErrorCode::invalid_argument,
                        "normalized state contains NaN or Inf",
                        {{"state", std::to_string(index)}});
        }
        state[index] = value;
    }
}

} // namespace wam::internal::policy
