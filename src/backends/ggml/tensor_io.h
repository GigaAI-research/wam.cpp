#pragma once

#include "ggml.h"

#include <vector>

namespace wam::internal::ggml_backend {

void set_f32(ggml_tensor * tensor, const std::vector<float> & values);
std::vector<float> get_f32(ggml_tensor * tensor);
void set_bf16(ggml_tensor * tensor,
              const std::vector<ggml_bf16_t> & values);
std::vector<ggml_bf16_t> get_bf16(ggml_tensor * tensor);

} // namespace wam::internal::ggml_backend
