#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace wam::internal::fastwam::debug {

bool enabled();
void dump(const std::string & name, const std::vector<float> & values,
          const std::vector<std::int64_t> & shape);
void dump(const std::string & name, const std::vector<ggml_bf16_t> & values,
          const std::vector<std::int64_t> & shape);

} // namespace wam::internal::fastwam::debug
