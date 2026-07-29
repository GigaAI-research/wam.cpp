#pragma once

#include "models/gwp05/state.h"

#include <vector>

namespace wam::internal::gwp05 {

bool build_prefix_cache(ExecutionState & state,
                        const std::vector<float> & model_state,
                        const std::vector<float> & reference,
                        const std::vector<float> & prompt);
bool run_unrolled_action_denoise(
    ExecutionState & state, std::vector<float> & action,
    const std::vector<float> & timesteps, const std::vector<float> & sigmas);
bool run_cached_action_step(
    ExecutionState & state, const std::vector<float> & action,
    const std::vector<float> & prompt, float timestep, float dt,
    bool upload_action, std::vector<float> * prediction,
    std::vector<float> * action_output, int step_index);

} // namespace wam::internal::gwp05
