#include "models/fastwam/scheduler.h"

#include "wam/error.h"

#include "ggml.h"

#include <cmath>
#include <string>

namespace wam::internal::fastwam {
namespace {

float shifted_sigma(float u, float shift) {
    return shift * u / (1.0f + (shift - 1.0f) * u);
}

float bf16_value(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

} // namespace

FlowSchedule make_inference_schedule(int steps, float shift,
                                     float train_timesteps) {
    if (steps <= 0) invalid("FastWAM inference step count must be positive",
                            "steps", std::to_string(steps));
    if (!std::isfinite(shift) || shift <= 0.0f) {
        invalid("FastWAM scheduler shift must be finite and positive",
                "shift", std::to_string(shift));
    }
    if (!std::isfinite(train_timesteps) || train_timesteps <= 0.0f) {
        invalid("FastWAM training timestep count must be finite and positive",
                "train_timesteps", std::to_string(train_timesteps));
    }

    // The donor builds sigma in F32, then casts both timestep and delta to the
    // BF16 action-latent dtype before denoising.
    FlowSchedule result;
    result.timesteps.resize(static_cast<std::size_t>(steps));
    result.deltas.resize(static_cast<std::size_t>(steps));
    float sigma = shifted_sigma(1.0f, shift);
    for (int index = 0; index < steps; ++index) {
        const float next_u = 1.0f - static_cast<float>(index + 1) /
            static_cast<float>(steps);
        const float next_sigma = shifted_sigma(next_u, shift);
        result.timesteps[static_cast<std::size_t>(index)] =
            bf16_value(sigma * train_timesteps);
        result.deltas[static_cast<std::size_t>(index)] =
            bf16_value(next_sigma - sigma);
        sigma = next_sigma;
    }
    return result;
}

} // namespace wam::internal::fastwam
