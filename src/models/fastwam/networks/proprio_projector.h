#pragma once

#include "ggml.h"

#include <vector>

namespace wam::internal::fastwam {

struct FastWamContract;

std::vector<ggml_bf16_t> project_proprio(
    const std::vector<float> & state,
    const FastWamContract & contract);

} // namespace wam::internal::fastwam
