#pragma once

#include "models/fastwam/contract.h"
#include "models/fastwam/state.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

struct VideoKvLayer {
    std::vector<ggml_bf16_t> key;
    std::vector<ggml_bf16_t> value;
};

struct VideoKvCache {
    std::size_t tokens = 0;
    std::vector<VideoKvLayer> layers;
};

std::vector<float> run_action_dit_step(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & action_input,
    const std::vector<ggml_bf16_t> & context,
    std::size_t context_tokens,
    const std::vector<std::int32_t> & context_mask,
    const VideoKvCache & video_cache,
    float timestep,
    const std::vector<std::int32_t> & positions);

} // namespace wam::internal::fastwam
