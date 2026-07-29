#pragma once

#include "models/gwp05/state.h"

#include <vector>

namespace wam::internal::gwp05 {

std::vector<float> run_vision_vae(ExecutionState & state,
                                  const PipelineInputsView & input);

} // namespace wam::internal::gwp05
