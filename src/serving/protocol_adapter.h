#pragma once

#include "wam/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam::serving {

enum class RpcMethod : std::uint32_t {
    get_model_info = 1,
    create_session = 2,
    predict = 3,
    reset_session = 4,
    close_session = 5,
};

enum class GrpcStatusCode : std::uint32_t {
    ok = 0,
    invalid_argument = 3,
    not_found = 5,
    resource_exhausted = 8,
    failed_precondition = 9,
    unimplemented = 12,
    internal = 13,
};

struct WireTensor {
    DType dtype = DType::unknown;
    std::vector<std::uint64_t> shape;
    std::string layout;
    ByteOrder byte_order = ByteOrder::unspecified;
    std::vector<std::uint8_t> data;
};

struct WireImage {
    std::string name;
    ImageEncoding encoding = ImageEncoding::unknown;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::uint64_t row_stride_bytes = 0;
    std::vector<std::uint8_t> data;
};

struct WireNamedTensor {
    std::string name;
    WireTensor tensor;
};

struct WireInputs {
    std::vector<WireImage> images;
    enum class LanguageKind { none, tokens, embedding } language_kind = LanguageKind::none;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
    WireTensor embedding;
    WireTensor embedding_attention_mask;
    WireTensor state;
    WireTensor noise;
    std::vector<WireNamedTensor> history;
};

struct WirePredictRequest {
    std::string session_id;
    std::uint64_t request_id = 0;
    WireInputs inputs;
};

std::vector<std::uint8_t> encode_predict_request(const WirePredictRequest & request);
WirePredictRequest decode_predict_request(const std::uint8_t * data, std::size_t size);

struct RpcResult {
    ErrorCode error_code = ErrorCode::ok;
    GrpcStatusCode grpc_status = GrpcStatusCode::ok;
    std::vector<std::uint8_t> payload;
    std::string message;

    explicit operator bool() const noexcept { return error_code == ErrorCode::ok; }
};

GrpcStatusCode grpc_status(ErrorCode code) noexcept;

class ProtocolService final {
public:
    explicit ProtocolService(const ModelOptions & options);
    ~ProtocolService();
    ProtocolService(const ProtocolService &) = delete;
    ProtocolService & operator=(const ProtocolService &) = delete;

    RpcResult call(RpcMethod method, const std::uint8_t * request, std::size_t size) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wam::serving
