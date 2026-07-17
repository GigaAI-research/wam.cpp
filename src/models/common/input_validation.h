#pragma once

#include "wam/types.h"

#include <string>
#include <vector>

namespace wam::internal {

std::size_t checked_numel(const std::vector<std::int64_t> & shape,
                          const std::string & field);

std::vector<float> copy_f32_tensor(
    const TensorView & tensor,
    const std::vector<std::vector<std::int64_t>> & allowed_shapes,
    const std::string & field);

} // namespace wam::internal
