#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace wam {

enum class DType : std::uint32_t {
    unknown = 0,
    u8,
    i32,
    f32,
    bf16,
};

enum class ByteOrder : std::uint32_t {
    unspecified = 0,
    little,
    big,
    not_applicable,
};

std::size_t dtype_size(DType dtype) noexcept;

struct TensorView {
    const void * data = nullptr;
    std::size_t byte_size = 0;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::string layout;
    ByteOrder byte_order = ByteOrder::not_applicable;

    bool empty() const noexcept;
};

struct NamedTensorView {
    std::string name;
    TensorView tensor;
};

enum class ImageEncoding : std::uint32_t {
    unknown = 0,
    rgb_u8,
    png,
    jpeg,
};

struct ImageView {
    std::string name;
    ImageEncoding encoding = ImageEncoding::unknown;
    const std::uint8_t * data = nullptr;
    std::size_t byte_size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::size_t row_stride_bytes = 0;
};

template <typename T>
struct ArrayView {
    const T * data = nullptr;
    std::size_t size = 0;

    ArrayView() = default;
    ArrayView(const T * values, std::size_t count) : data(values), size(count) {}
    explicit ArrayView(const std::vector<T> & values)
        : data(values.data()), size(values.size()) {}
    ArrayView(std::vector<T> &&) = delete;
    bool empty() const noexcept { return size == 0; }
};

struct TokenInput {
    ArrayView<std::int32_t> token_ids;
    ArrayView<std::int32_t> attention_mask;
};

struct EmbeddingInput {
    TensorView embedding;
    TensorView attention_mask;
};

using LanguageInput = std::variant<std::monostate, TokenInput, EmbeddingInput>;

struct Observation {
    std::vector<ImageView> images;
    LanguageInput language;
    TensorView state;
    TensorView action_noise;
    std::vector<NamedTensorView> history;
};

} // namespace wam
