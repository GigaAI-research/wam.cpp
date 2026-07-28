#pragma once

#include "ggml.h"

#include <vector>

namespace wam::internal::ggml_backend {

void set_f32(ggml_tensor * tensor, const std::vector<float> & values);
std::vector<float> get_f32(ggml_tensor * tensor);

} // namespace wam::internal::ggml_backend
