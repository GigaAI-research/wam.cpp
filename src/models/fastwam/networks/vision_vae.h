#pragma once

#include "models/fastwam/state.h"
#include "models/fastwam/contract.h"

#include "ggml.h"

#include <vector>

namespace wam::internal::fastwam {

std::vector<ggml_bf16_t> encode_first_frame(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<float> & patchified_pixels);

} // namespace wam::internal::fastwam
