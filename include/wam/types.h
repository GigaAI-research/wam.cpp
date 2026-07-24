#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace wam {

enum class ErrorCode : std::uint32_t {
    ok = 0,
    invalid_argument,
    not_found,
    unsupported,
    incompatible_artifact,
    resource_exhausted,
    failed_precondition,
    inference_failed,
    internal,
};

struct ErrorDetail {
    std::string field;
    std::string reason;
};

struct Status {
    ErrorCode code = ErrorCode::ok;
    std::string message;
    std::vector<ErrorDetail> details;

    explicit operator bool() const noexcept;
    static Status success();
};

class Error final : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message,
          std::vector<ErrorDetail> details = {});
    ErrorCode code() const noexcept;
    const std::vector<ErrorDetail> & details() const noexcept;

private:
    ErrorCode code_;
    std::vector<ErrorDetail> details_;
};

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

struct Tensor {
    std::vector<std::uint8_t> data;
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

struct NamedTensor {
    std::string name;
    Tensor tensor;
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

    ArrayView();
    ArrayView(const T * values, std::size_t count);
    explicit ArrayView(const std::vector<T> & values);
    ArrayView(std::vector<T> &&) = delete;
    bool empty() const noexcept;
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

struct Inputs {
    std::vector<ImageView> images;
    LanguageInput language;
    TensorView state;
    TensorView action_noise;
    std::vector<NamedTensorView> history;
};

enum class Backend : std::uint32_t {
    unknown = 0,
    automatic,
    cuda,
    cpu_metadata,
};

enum class ComputePrecision : std::uint32_t {
    unknown = 0,
    automatic,
    f32,
    f16,
    bf16,
    fp8_e4m3,
    fp8_e5m2,
    int8,
};

enum class LanguageRuntimeMode : std::uint32_t {
    automatic = 0,
    tokens,
    external_embedding,
};

struct FixedPrompt {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
};

struct ModelOptions {
    std::string artifact_path;
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    std::int32_t device_index = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::optional<FixedPrompt> fixed_prompt;
};

struct SessionOptions {
    bool enable_prefix_cache = true;
    std::uint64_t random_seed = 0;
};

struct PhaseTiming {
    std::string name;
    double milliseconds = 0.0;
};

struct Stats {
    double preprocess_milliseconds = 0.0;
    double model_milliseconds = 0.0;
    double model_vision_milliseconds = 0.0;
    double model_text_milliseconds = 0.0;
    double model_prefill_milliseconds = 0.0;
    double model_decode_milliseconds = 0.0;
    double postprocess_milliseconds = 0.0;
    double total_milliseconds = 0.0;

    std::vector<PhaseTiming> model_timings;
    std::uint64_t peak_device_memory_bytes = 0;
};

struct Prediction {
    Tensor action;
    std::vector<NamedTensor> auxiliary;
    Stats stats;
};

struct Capabilities {
    bool action = false;
    bool auxiliary_outputs = false;
    bool raw_images = false;
    bool token_input = false;
    bool precomputed_embedding = false;
    bool explicit_action_noise = false;
    bool batch_inference = false;
    bool concurrent_sessions = false;
    bool arbitrary_token_input = false;
    bool fixed_token_input = false;
    std::vector<Backend> backends;
    std::vector<ComputePrecision> compute_precisions;
};

struct RuntimeComponentInfo {
    std::string name;
    bool loaded = false;
    std::uint64_t device_bytes = 0;
    double load_milliseconds = 0.0;
    double unload_milliseconds = 0.0;
};

struct ArtifactComponentInfo {
    std::string name;
    DType source_dtype = DType::unknown;
    DType stored_dtype = DType::unknown;
    std::uint64_t tensor_count = 0;
    std::uint64_t elements = 0;
    std::uint64_t source_bytes = 0;
    std::uint64_t stored_bytes = 0;
};

struct ModelInfo {
    std::string architecture;
    std::string artifact_path;
    std::string artifact_policy;
    std::uint64_t artifact_bytes = 0;
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::unknown;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;
    Capabilities capabilities;
    std::vector<ArtifactComponentInfo> artifact_components;
    std::vector<RuntimeComponentInfo> runtime_components;
};

} // namespace wam
