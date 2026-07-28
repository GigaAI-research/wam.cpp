#include "backends/ggml/graph_ops.h"

namespace wam::internal::ggml_backend {

ggml_tensor * as_f32(ggml_context * context, ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_F32
        ? tensor
        : ggml_cast(context, tensor, GGML_TYPE_F32);
}

ggml_tensor * as_bf16(ggml_context * context, ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_BF16
        ? tensor
        : ggml_cast(context, tensor, GGML_TYPE_BF16);
}

} // namespace wam::internal::ggml_backend
