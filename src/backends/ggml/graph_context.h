#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>

namespace wam::internal::ggml_backend {

class GraphContext {
public:
    GraphContext() = default;
    explicit GraphContext(std::size_t metadata_bytes);
    ~GraphContext();
    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;

    void initialize(std::size_t metadata_bytes);
    ggml_context * get() const noexcept;
    void allocate(ggml_cgraph * graph, ggml_backend_buffer_type_t type);
    ggml_gallocr_t allocator() const noexcept;

    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
};

} // namespace wam::internal::ggml_backend
