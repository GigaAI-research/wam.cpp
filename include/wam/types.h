#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wam {

enum class ErrorCode : std::uint32_t {
    ok = 0,
    invalid_argument = 1,
    not_found = 2,
    unsupported = 3,
    incompatible_artifact = 4,
    resource_exhausted = 5,
    failed_precondition = 6,
    inference_failed = 7,
    internal = 8,
};

struct ErrorDetail {
    std::string field;
    std::string reason;
};

struct Status {
    ErrorCode code = ErrorCode::ok;
    std::string message;
    std::vector<ErrorDetail> details;

    explicit operator bool() const noexcept { return code == ErrorCode::ok; }
    static Status success() { return {}; }
};

class Error final : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message,
          std::vector<ErrorDetail> details = {})
        : std::runtime_error(std::move(message)), code_(code), details_(std::move(details)) {}

    ErrorCode code() const noexcept { return code_; }
    const std::vector<ErrorDetail> & details() const noexcept { return details_; }

private:
    ErrorCode code_;
    std::vector<ErrorDetail> details_;
};

enum class DType : std::uint32_t {
    unknown = 0,
    u8 = 1,
    i32 = 2,
    f32 = 3,
    bf16 = 4,
};

enum class ByteOrder : std::uint32_t {
    unspecified = 0,
    little = 1,
    big = 2,
    not_applicable = 3,
};

inline constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::u8: return 1;
        case DType::i32:
        case DType::f32: return 4;
        case DType::bf16: return 2;
        case DType::unknown: return 0;
    }
    return 0;
}

// Payload memory is borrowed from the caller for the duration of predict().
struct TensorView {
    const void * data = nullptr;
    std::size_t byte_size = 0;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::string layout;
    ByteOrder byte_order = ByteOrder::not_applicable;

    bool empty() const noexcept { return data == nullptr || byte_size == 0; }
};

// Tensor owns its payload and is safe after predict() returns.
struct Tensor {
    std::vector<std::uint8_t> data;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::string layout;
    ByteOrder byte_order = ByteOrder::not_applicable;

    bool empty() const noexcept { return data.empty(); }
};

struct NamedTensor {
    std::string name;
    Tensor tensor;
};

struct NamedTensorView {
    std::string name;
    TensorView tensor;
};

enum class ImageEncoding : std::uint32_t {
    unknown = 0,
    rgb_u8 = 1,
    png = 2,
    jpeg = 3,
};

// Encoded bytes or packed RGB bytes are borrowed during predict().
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
    explicit ArrayView(const std::vector<T> & values) : data(values.data()), size(values.size()) {}
    ArrayView(std::vector<T> &&) = delete;
    bool empty() const noexcept { return data == nullptr || size == 0; }
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
    TensorView noise;
    std::vector<NamedTensorView> history;
};

enum class Backend : std::uint32_t {
    unknown = 0,
    automatic = 1,
    cuda = 2,
    cpu_metadata = 3,
};

enum class Precision : std::uint32_t {
    unknown = 0,
    f32_reference = 1,
    bf16_latency = 2,
};

enum class LanguageEncoderPolicy : std::uint32_t {
    resident = 0,
    fixed = 1,
    external_embedding = 2,
};

struct FixedPrompt {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
};

struct ModelOptions {
    std::string artifact_path;
    Backend backend = Backend::automatic;
    Precision precision = Precision::f32_reference;
    std::int32_t device_index = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageEncoderPolicy language_encoder_policy = LanguageEncoderPolicy::resident;
    std::optional<FixedPrompt> fixed_prompt;
};

struct SessionOptions {
    bool enable_prefix_cache = true;
};

struct PhaseTiming {
    std::string name;
    double milliseconds = 0.0;
};

struct Stats {
    double preprocess_milliseconds = 0.0;
    double vae_milliseconds = 0.0;
    double text_encoder_milliseconds = 0.0;
    double prompt_projection_milliseconds = 0.0;
    double prefix_milliseconds = 0.0;
    double denoise_milliseconds = 0.0;
    double postprocess_milliseconds = 0.0;
    double total_milliseconds = 0.0;
    std::vector<double> denoise_step_milliseconds;
    std::vector<PhaseTiming> additional_timings;
    bool prompt_cache_hit = false;
    bool projected_prompt_cache_hit = false;
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
    bool explicit_noise = false;
    bool batch_inference = false;
    bool concurrent_sessions = false;
    bool arbitrary_token_input = false;
    bool fixed_token_input = false;
    std::vector<Backend> backends;
    std::vector<Precision> precisions;
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
    std::string artifact_sha256;
    std::uint64_t artifact_bytes = 0;
    Backend backend = Backend::automatic;
    Precision precision = Precision::f32_reference;
    LanguageEncoderPolicy language_encoder_policy = LanguageEncoderPolicy::resident;
    bool text_encoder_resident = false;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;
    Capabilities capabilities;
    std::vector<ArtifactComponentInfo> artifact_components;
    std::vector<RuntimeComponentInfo> runtime_components;
};

inline constexpr const char * kImageHigh = "camera_high";
inline constexpr const char * kImageLeftWrist = "camera_left_wrist";
inline constexpr const char * kImageRightWrist = "camera_right_wrist";
inline constexpr const char * kGwp05ReferenceLatent = "gwp05.reference_latent";

} // namespace wam
