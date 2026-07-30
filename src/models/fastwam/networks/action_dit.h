#pragma once

#include "models/fastwam/contract.h"
#include "models/fastwam/graphs.h"
#include "models/fastwam/scheduler.h"
#include "models/fastwam/state.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

std::vector<float> run_unrolled_action_denoise(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & action_input,
    const std::vector<ggml_bf16_t> & context,
    std::size_t context_tokens,
    const std::vector<std::int32_t> & context_mask,
    const DeviceVideoKvCache & video_cache,
    const FlowSchedule & schedule,
    const std::vector<std::int32_t> & positions);

} // namespace wam::internal::fastwam
