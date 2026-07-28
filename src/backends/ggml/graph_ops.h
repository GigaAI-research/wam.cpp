#pragma once

#include "ggml.h"

namespace wam::internal::ggml_backend {

ggml_tensor * as_f32(ggml_context * context, ggml_tensor * tensor);
ggml_tensor * as_bf16(ggml_context * context, ggml_tensor * tensor);

} // namespace wam::internal::ggml_backend
