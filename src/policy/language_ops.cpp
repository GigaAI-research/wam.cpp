#include "policy/language_ops.h"

#include "policy/tensor_input.h"
#include "wam/error.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace wam::internal::policy {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

void validate_finite(const TensorView & tensor) {
    const std::size_t elements =
        checked_numel(tensor.shape, "language.embedding");
    const auto * bytes = static_cast<const std::uint8_t *>(tensor.data);
    if (tensor.dtype == DType::f32) {
        for (std::size_t index = 0; index < elements; ++index) {
            float value = 0.0F;
            std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
            if (!std::isfinite(value)) {
                invalid("prompt embedding contains NaN or Inf",
                        "language.embedding", std::to_string(index));
            }
        }
        return;
    }
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t value = 0;
        std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
        if ((value & 0x7F80U) == 0x7F80U) {
            invalid("prompt embedding contains NaN or Inf",
                    "language.embedding", std::to_string(index));
        }
    }
}

void validate_mask(const std::vector<std::int32_t> & mask) {
    bool padding_started = false;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index] != 0 && mask[index] != 1) {
            invalid("embedding attention mask must be binary",
                    "language.attention_mask", std::to_string(index));
        }
        if (mask[index] == 0) padding_started = true;
        if (padding_started && mask[index] != 0) {
            invalid("embedding attention mask must be contiguous",
                    "language.attention_mask", std::to_string(index));
        }
    }
}

} // namespace

PreparedEmbedding prepare_embedding_input(const EmbeddingInput & input,
                                          const LanguageSpec & spec,
                                          std::size_t expected_width) {
    const TensorView & embedding = input.embedding;
    if (embedding.data == nullptr || embedding.shape.size() != 2 ||
        embedding.shape[0] <= 0 || embedding.shape[1] <= 0 ||
        static_cast<std::size_t>(embedding.shape[1]) != expected_width ||
        static_cast<std::size_t>(embedding.shape[0]) > spec.max_tokens ||
        embedding.layout != "T,D" ||
        embedding.byte_order != ByteOrder::little ||
        (embedding.dtype != DType::f32 && embedding.dtype != DType::bf16)) {
        invalid("prompt embedding contract is invalid", "language.embedding",
                "expected little-endian F32/BF16 [T,D] with layout T,D");
    }
    const std::size_t elements =
        checked_numel(embedding.shape, "language.embedding");
    const std::size_t width = dtype_size(embedding.dtype);
    if (width == 0 ||
        elements > std::numeric_limits<std::size_t>::max() / width ||
        embedding.byte_size != elements * width) {
        invalid("prompt embedding byte size is invalid", "language.embedding",
                "payload size mismatch");
    }
    validate_finite(embedding);

    PreparedEmbedding prepared;
    const std::size_t tokens = static_cast<std::size_t>(embedding.shape[0]);
    if (input.attention_mask.empty()) {
        if (spec.attention_mask_required) {
            invalid("embedding attention mask is required",
                    "language.attention_mask", "missing");
        }
        prepared.attention_mask.assign(tokens, 1);
    } else {
        prepared.attention_mask = copy_i32_tensor(
            input.attention_mask, {static_cast<std::int64_t>(tokens)},
            "language.attention_mask");
    }
    validate_mask(prepared.attention_mask);

    prepared.embedding.dtype = embedding.dtype;
    prepared.embedding.shape = embedding.shape;
    prepared.embedding.layout = embedding.layout;
    prepared.embedding.byte_order = embedding.byte_order;
    prepared.embedding.data.resize(embedding.byte_size);
    std::memcpy(prepared.embedding.data.data(), embedding.data,
                embedding.byte_size);
    return prepared;
}

} // namespace wam::internal::policy
