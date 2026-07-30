#pragma once

#include "backends/ggml/graph_context.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

class DeviceVideoKvCache final {
public:
    DeviceVideoKvCache() = default;
    ~DeviceVideoKvCache();
    DeviceVideoKvCache(const DeviceVideoKvCache &) = delete;
    DeviceVideoKvCache & operator=(const DeviceVideoKvCache &) = delete;

    void initialize(ggml_backend_t backend, std::size_t layers,
                    std::size_t tokens, std::size_t heads,
                    std::size_t head_dim);
    bool matches(std::size_t layers, std::size_t tokens,
                 std::size_t heads, std::size_t head_dim) const noexcept;
    ggml_tensor * key(std::size_t layer) const;
    ggml_tensor * value(std::size_t layer) const;
    std::size_t layers() const noexcept;
    std::size_t tokens() const noexcept;
    std::uint64_t bytes() const noexcept;

private:
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::size_t tokens_ = 0;
    std::size_t heads_ = 0;
    std::size_t head_dim_ = 0;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
};

struct UnrolledActionGraph final : ggml_backend::GraphContext {
    UnrolledActionGraph();
    ~UnrolledActionGraph();

    ggml_cgraph * graph = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * context_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * context_attention_mask = nullptr;
    std::vector<ggml_tensor *> frequency_inputs;
    std::vector<ggml_tensor *> delta_inputs;
    std::vector<ggml_tensor *> debug_velocities;
    std::vector<ggml_tensor *> debug_action_states;
    ggml_tensor * action_output = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
    std::size_t context_tokens = 0;
    std::size_t video_tokens = 0;
    std::size_t steps = 0;

    std::uint64_t bytes() const noexcept;
};

} // namespace wam::internal::fastwam
