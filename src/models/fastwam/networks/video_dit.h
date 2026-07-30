#pragma once

#include "models/fastwam/contract.h"
#include "models/fastwam/graphs.h"
#include "models/fastwam/state.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

void prefill_video_cache(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & latent,
    const std::vector<ggml_bf16_t> & context,
    const std::vector<std::int32_t> & context_mask,
    std::size_t context_tokens, DeviceVideoKvCache & cache);

} // namespace wam::internal::fastwam
