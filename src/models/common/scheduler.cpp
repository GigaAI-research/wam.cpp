#include "scheduler.h"

#include "wam/types.h"

#include <cmath>
#include <string>

namespace wam::internal {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

} // namespace

float FlowMatchEulerSchedule::delta_sigma(std::size_t step) const {
    if (sigmas.size() != timesteps.size() + 1 || step >= timesteps.size()) {
        invalid("flow-match scheduler step is out of range", "step", std::to_string(step));
    }
    return sigmas[step + 1] - sigmas[step];
}

FlowMatchEulerSchedule make_flow_match_euler_schedule(int steps, float shift) {
    if (steps <= 0) invalid("flow-match step count must be positive", "steps", std::to_string(steps));
    if (!std::isfinite(shift) || shift <= 0.0f) {
        invalid("flow-match shift must be finite and positive", "shift", std::to_string(shift));
    }

    FlowMatchEulerSchedule schedule;
    schedule.timesteps.resize(static_cast<std::size_t>(steps));
    schedule.sigmas.resize(static_cast<std::size_t>(steps + 1));

    // Diffusers derives the endpoint from its float32 training schedule, then
    // performs linspace and the shift transform in float64.
    const float training_sigma = 1.0f / 1000.0f;
    const float shifted_sigma_min =
        shift * training_sigma / (1.0f + (shift - 1.0f) * training_sigma);
    const double minimum_timestep = static_cast<double>(shifted_sigma_min) * 1000.0;
    for (int step = 0; step < steps; ++step) {
        const double unshifted_t = steps == 1
            ? 1000.0
            : 1000.0 + (minimum_timestep - 1000.0) * step / (steps - 1);
        const double sigma = unshifted_t / 1000.0;
        const double shifted = static_cast<double>(shift) * sigma /
            (1.0 + (static_cast<double>(shift) - 1.0) * sigma);
        schedule.sigmas[static_cast<std::size_t>(step)] = static_cast<float>(shifted);
        schedule.timesteps[static_cast<std::size_t>(step)] =
            schedule.sigmas[static_cast<std::size_t>(step)] * 1000.0f;
    }
    schedule.sigmas[static_cast<std::size_t>(steps)] = 0.0f;
    return schedule;
}

void flow_match_euler_step(std::vector<float> & action,
                           const std::vector<float> & velocity,
                           float delta_sigma) {
    if (action.size() != velocity.size()) {
        invalid("flow-match action and velocity sizes differ", "velocity",
                std::to_string(velocity.size()));
    }
    if (!std::isfinite(delta_sigma)) {
        invalid("flow-match delta sigma must be finite", "delta_sigma", "non-finite");
    }
    for (std::size_t index = 0; index < action.size(); ++index) {
        if (!std::isfinite(action[index]) || !std::isfinite(velocity[index])) {
            invalid("flow-match inputs must be finite", "action_or_velocity",
                    std::to_string(index));
        }
        action[index] += delta_sigma * velocity[index];
    }
}

} // namespace wam::internal
