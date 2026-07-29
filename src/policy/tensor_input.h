#pragma once

#include "wam/observation.h"

#include <string>
#include <vector>

namespace wam::internal::policy {

std::size_t checked_numel(const std::vector<std::int64_t> & shape,
                          const std::string & field);
std::vector<float> copy_f32_tensor(
    const TensorView & tensor,
    const std::vector<std::int64_t> & expected_shape,
    const std::string & field);
std::vector<std::int32_t> copy_i32_tensor(
    const TensorView & tensor, const std::vector<std::int64_t> & shape,
    const std::string & field);

} // namespace wam::internal::policy
