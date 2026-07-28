#include "backends/ggml/graph_context.h"

#include "wam/error.h"

namespace wam::internal::ggml_backend {

GraphContext::GraphContext(std::size_t metadata_bytes) {
    initialize(metadata_bytes);
}

void GraphContext::initialize(std::size_t metadata_bytes) {
    if (metadata_bytes == 0) {
        throw Error(ErrorCode::invalid_argument,
                    "graph metadata size must be positive");
    }
    ggml_init_params parameters{};
    parameters.mem_size = metadata_bytes;
    parameters.no_alloc = true;
    if (ctx != nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "GGML graph context is already initialized");
    }
    ctx = ggml_init(parameters);
    if (ctx == nullptr) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate GGML graph metadata context");
    }
}

GraphContext::~GraphContext() {
    if (alloc != nullptr) ggml_gallocr_free(alloc);
    if (ctx != nullptr) ggml_free(ctx);
}

ggml_context * GraphContext::get() const noexcept { return ctx; }

void GraphContext::allocate(ggml_cgraph * graph,
                            ggml_backend_buffer_type_t type) {
    if (graph == nullptr || type == nullptr || alloc != nullptr) {
        throw Error(ErrorCode::invalid_argument,
                    "GGML graph allocation arguments are invalid");
    }
    alloc = ggml_gallocr_new(type);
    if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph)) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate GGML graph");
    }
}

ggml_gallocr_t GraphContext::allocator() const noexcept { return alloc; }

} // namespace wam::internal::ggml_backend
