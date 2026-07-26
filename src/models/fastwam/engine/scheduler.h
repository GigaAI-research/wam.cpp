#pragma once

#include <cstddef>
#include <vector>

namespace wam::internal::fastwam {

struct FlowSchedule {
    std::vector<float> timesteps;
    std::vector<float> deltas;

    std::size_t steps() const noexcept { return timesteps.size(); }
};

FlowSchedule make_inference_schedule(int steps, float shift,
                                     float train_timesteps = 1000.0f);

} // namespace wam::internal::fastwam
