#include "models/fastwam/graphs.h"

#include "wam/error.h"

#include <limits>
#include <string>

namespace wam::internal::fastwam {

DeviceVideoKvCache::~DeviceVideoKvCache() {
    if (buffer_ != nullptr) ggml_backend_buffer_free(buffer_);
    if (context_ != nullptr) ggml_free(context_);
}

void DeviceVideoKvCache::initialize(
    ggml_backend_t backend, std::size_t layer_count, std::size_t token_count,
    std::size_t head_count, std::size_t head_dimension) {
    if (matches(layer_count, token_count, head_count, head_dimension)) return;
    if (buffer_ != nullptr || context_ != nullptr || backend == nullptr ||
        layer_count == 0 || token_count == 0 || head_count == 0 ||
        head_dimension == 0 ||
        layer_count > (std::numeric_limits<std::size_t>::max() /
                           ggml_tensor_overhead() - 32) / 2) {
        throw Error(ErrorCode::failed_precondition,
                    "FastWAM device K/V cache geometry changed");
    }

    ggml_init_params parameters{};
    parameters.mem_size =
        (2 * layer_count + 32) * ggml_tensor_overhead();
    parameters.no_alloc = true;
    context_ = ggml_init(parameters);
    if (context_ == nullptr) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate FastWAM K/V cache metadata");
    }

    tokens_ = token_count;
    heads_ = head_count;
    head_dim_ = head_dimension;
    keys_.reserve(layer_count);
    values_.reserve(layer_count);
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        ggml_tensor * key = ggml_new_tensor_3d(
            context_, GGML_TYPE_BF16, static_cast<std::int64_t>(head_dim_),
            static_cast<std::int64_t>(heads_),
            static_cast<std::int64_t>(tokens_));
        ggml_tensor * value = ggml_new_tensor_3d(
            context_, GGML_TYPE_BF16, static_cast<std::int64_t>(head_dim_),
            static_cast<std::int64_t>(heads_),
            static_cast<std::int64_t>(tokens_));
        if (key == nullptr || value == nullptr) {
            keys_.clear();
            values_.clear();
            ggml_free(context_);
            context_ = nullptr;
            tokens_ = heads_ = head_dim_ = 0;
            throw Error(ErrorCode::resource_exhausted,
                        "cannot define FastWAM device K/V cache tensors");
        }
        keys_.push_back(key);
        values_.push_back(value);
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend);
    if (buffer_ == nullptr) {
        keys_.clear();
        values_.clear();
        ggml_free(context_);
        context_ = nullptr;
        tokens_ = heads_ = head_dim_ = 0;
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate FastWAM device K/V cache");
    }
}

bool DeviceVideoKvCache::matches(
    std::size_t layer_count, std::size_t token_count,
    std::size_t head_count, std::size_t head_dimension) const noexcept {
    return buffer_ != nullptr && keys_.size() == layer_count &&
        values_.size() == layer_count && tokens_ == token_count &&
        heads_ == head_count && head_dim_ == head_dimension;
}

ggml_tensor * DeviceVideoKvCache::key(std::size_t layer) const {
    if (layer >= keys_.size()) {
        throw Error(ErrorCode::invalid_argument,
                    "FastWAM K/V cache key layer is out of range",
                    {{"layer", std::to_string(layer)}});
    }
    return keys_[layer];
}

ggml_tensor * DeviceVideoKvCache::value(std::size_t layer) const {
    if (layer >= values_.size()) {
        throw Error(ErrorCode::invalid_argument,
                    "FastWAM K/V cache value layer is out of range",
                    {{"layer", std::to_string(layer)}});
    }
    return values_[layer];
}

std::size_t DeviceVideoKvCache::layers() const noexcept {
    return keys_.size();
}

std::size_t DeviceVideoKvCache::tokens() const noexcept { return tokens_; }

std::uint64_t DeviceVideoKvCache::bytes() const noexcept {
    return buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(buffer_);
}

UnrolledActionGraph::UnrolledActionGraph()
    : GraphContext(512u * 1024u * 1024u) {}

UnrolledActionGraph::~UnrolledActionGraph() {
    if (action_input_buffer != nullptr) {
        ggml_backend_buffer_free(action_input_buffer);
    }
}

std::uint64_t UnrolledActionGraph::bytes() const noexcept {
    std::uint64_t result = action_input_buffer == nullptr
        ? 0 : ggml_backend_buffer_get_size(action_input_buffer);
    if (allocator() != nullptr) {
        result += ggml_gallocr_get_buffer_size(allocator(), 0);
    }
    return result;
}

} // namespace wam::internal::fastwam
