#pragma once

#include <cstddef>
#include <vector>

namespace wam::internal {

struct FlowMatchEulerSchedule {
    std::vector<float> timesteps;
    std::vector<float> sigmas;

    std::size_t steps() const noexcept { return timesteps.size(); }
    float delta_sigma(std::size_t step) const;
};

FlowMatchEulerSchedule make_flow_match_euler_schedule(int steps, float shift);

void flow_match_euler_step(std::vector<float> & action,
                           const std::vector<float> & velocity,
                           float delta_sigma);

} // namespace wam::internal
