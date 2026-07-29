#pragma once

#include "models/gwp05/state.h"

#include <vector>

namespace wam::internal::gwp05 {

bool validate_prompt_mask(const PipelineInputsView & input, int & valid_tokens);
std::vector<float> run_umt5(ExecutionState & state,
                            const PipelineInputsView & input);

} // namespace wam::internal::gwp05
