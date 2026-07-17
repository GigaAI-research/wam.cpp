#include "protocol_adapter.h"

#include "wam/version.h"
#include "wam/wam.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>

namespace wam::serving {
namespace {

[[noreturn]] void invalid_wire(const std::string & message,
                               const std::string & field = "request") {
    throw Error(ErrorCode::invalid_argument, message, {{field, "invalid protobuf wire data"}});
}

class Reader {
public:
    Reader(const std::uint8_t * data, std::size_t size)
        : current_(data), end_(data == nullptr ? data : data + size) {
        if (data == nullptr && size != 0) invalid_wire("null protobuf payload");
    }

    bool done() const noexcept { return current_ == end_; }

    std::uint64_t varint() {
        std::uint64_t result = 0;
        for (unsigned shift = 0; shift < 64; shift += 7) {
            if (done()) invalid_wire("truncated protobuf varint");
            const std::uint8_t byte = *current_++;
            result |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
            if ((byte & 0x80U) == 0) return result;
        }
        invalid_wire("protobuf varint overflows uint64");
    }

    std::pair<std::uint32_t, std::uint32_t> tag() {
        const std::uint64_t value = varint();
        if ((value >> 3U) == 0 || (value >> 3U) > std::numeric_limits<std::uint32_t>::max()) {
            invalid_wire("invalid protobuf field number");
        }
        return {static_cast<std::uint32_t>(value >> 3U),
                static_cast<std::uint32_t>(value & 7U)};
    }

    Reader message() {
        const std::uint64_t length = varint();
        if (length > remaining()) invalid_wire("truncated protobuf message");
        Reader value(current_, static_cast<std::size_t>(length));
        current_ += length;
        return value;
    }

    std::vector<std::uint8_t> bytes() {
        Reader value = message();
        return {value.current_, value.end_};
    }

    std::string string() {
        const std::vector<std::uint8_t> value = bytes();
        return {reinterpret_cast<const char *>(value.data()), value.size()};
    }

    std::uint64_t fixed64() {
        if (remaining() < sizeof(std::uint64_t)) invalid_wire("truncated fixed64");
        std::uint64_t value = 0;
        for (unsigned index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(current_[index]) << (8U * index);
        }
        current_ += 8;
        return value;
    }

    void skip(std::uint32_t wire) {
        if (wire == 0) { (void) varint(); return; }
        if (wire == 1) { advance(8); return; }
        if (wire == 2) { Reader ignored = message(); (void) ignored; return; }
        if (wire == 5) { advance(4); return; }
        invalid_wire("unsupported protobuf wire type");
    }

private:
    std::uint64_t remaining() const noexcept {
        return current_ == nullptr ? 0 : static_cast<std::uint64_t>(end_ - current_);
    }
    void advance(std::size_t size) {
        if (size > remaining()) invalid_wire("truncated protobuf field");
        current_ += size;
    }

    const std::uint8_t * current_ = nullptr;
    const std::uint8_t * end_ = nullptr;
};

class Writer {
public:
    void varint(std::uint64_t value) {
        while (value >= 0x80U) {
            data_.push_back(static_cast<std::uint8_t>(value) | 0x80U);
            value >>= 7U;
        }
        data_.push_back(static_cast<std::uint8_t>(value));
    }

    void tag(std::uint32_t field, std::uint32_t wire) {
        varint((static_cast<std::uint64_t>(field) << 3U) | wire);
    }

    void enum_field(std::uint32_t field, std::uint32_t value) {
        if (value == 0) return;
        tag(field, 0); varint(value);
    }
    void u32(std::uint32_t field, std::uint32_t value) {
        if (value == 0) return;
        tag(field, 0); varint(value);
    }
    void u64(std::uint32_t field, std::uint64_t value) {
        if (value == 0) return;
        tag(field, 0); varint(value);
    }
    void boolean(std::uint32_t field, bool value) {
        if (!value) return;
        tag(field, 0); varint(1);
    }
    void string(std::uint32_t field, const std::string & value) {
        if (value.empty()) return;
        bytes(field, reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
    }
    void bytes(std::uint32_t field, const std::vector<std::uint8_t> & value) {
        if (value.empty()) return;
        bytes(field, value.data(), value.size());
    }
    void message(std::uint32_t field, const std::vector<std::uint8_t> & value,
                 bool include_empty = true) {
        if (value.empty() && !include_empty) return;
        tag(field, 2); varint(value.size());
        data_.insert(data_.end(), value.begin(), value.end());
    }
    void packed_u64(std::uint32_t field, const std::vector<std::uint64_t> & values) {
        if (values.empty()) return;
        Writer packed;
        for (std::uint64_t value : values) packed.varint(value);
        message(field, packed.take());
    }
    void packed_i32(std::uint32_t field, const std::vector<std::int32_t> & values) {
        if (values.empty()) return;
        Writer packed;
        for (std::int32_t value : values) {
            packed.varint(static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
        }
        message(field, packed.take());
    }
    void fixed64_field(std::uint32_t field, double value) {
        if (value == 0.0) return;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        tag(field, 1);
        for (unsigned index = 0; index < 8; ++index) {
            data_.push_back(static_cast<std::uint8_t>(bits >> (8U * index)));
        }
    }
    void packed_double(std::uint32_t field, const std::vector<double> & values) {
        if (values.empty()) return;
        Writer packed;
        for (double value : values) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (unsigned index = 0; index < 8; ++index) {
                packed.data_.push_back(static_cast<std::uint8_t>(bits >> (8U * index)));
            }
        }
        message(field, packed.take());
    }

    std::vector<std::uint8_t> take() { return std::move(data_); }

private:
    void bytes(std::uint32_t field, const std::uint8_t * data, std::size_t size) {
        tag(field, 2); varint(size); data_.insert(data_.end(), data, data + size);
    }
    std::vector<std::uint8_t> data_;
};

void need_wire(std::uint32_t actual, std::uint32_t expected, const char * field) {
    if (actual != expected) invalid_wire(std::string("wrong wire type for ") + field, field);
}

template <typename Value>
void parse_packed_varints(Reader & reader, std::uint32_t wire, std::vector<Value> & output,
                          const char * field) {
    if (wire == 0) {
        output.push_back(static_cast<Value>(reader.varint()));
        return;
    }
    need_wire(wire, 2, field);
    Reader packed = reader.message();
    while (!packed.done()) output.push_back(static_cast<Value>(packed.varint()));
}

WireTensor parse_tensor(Reader reader) {
    WireTensor result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        switch (field) {
            case 1: need_wire(wire, 0, "tensor.dtype");
                    result.dtype = static_cast<DType>(reader.varint()); break;
            case 2: parse_packed_varints(reader, wire, result.shape, "tensor.shape"); break;
            case 3: need_wire(wire, 2, "tensor.layout"); result.layout = reader.string(); break;
            case 4: need_wire(wire, 0, "tensor.byte_order");
                    result.byte_order = static_cast<ByteOrder>(reader.varint()); break;
            case 5: need_wire(wire, 2, "tensor.data"); result.data = reader.bytes(); break;
            default: reader.skip(wire);
        }
    }
    return result;
}

WireImage parse_image(Reader reader) {
    WireImage result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        switch (field) {
            case 1: need_wire(wire, 2, "image.name"); result.name = reader.string(); break;
            case 2: need_wire(wire, 0, "image.encoding");
                    result.encoding = static_cast<ImageEncoding>(reader.varint()); break;
            case 3: need_wire(wire, 0, "image.width"); result.width = reader.varint(); break;
            case 4: need_wire(wire, 0, "image.height"); result.height = reader.varint(); break;
            case 5: need_wire(wire, 0, "image.channels"); result.channels = reader.varint(); break;
            case 6: need_wire(wire, 0, "image.row_stride_bytes");
                    result.row_stride_bytes = reader.varint(); break;
            case 7: need_wire(wire, 2, "image.data"); result.data = reader.bytes(); break;
            default: reader.skip(wire);
        }
    }
    return result;
}

WireNamedTensor parse_named_tensor(Reader reader) {
    WireNamedTensor result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) { need_wire(wire, 2, "named_tensor.name"); result.name = reader.string(); }
        else if (field == 2) { need_wire(wire, 2, "named_tensor.tensor");
                               result.tensor = parse_tensor(reader.message()); }
        else reader.skip(wire);
    }
    return result;
}

void parse_tokens(Reader reader, WireInputs & result) {
    result.language_kind = WireInputs::LanguageKind::tokens;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) parse_packed_varints(reader, wire, result.token_ids, "tokens.token_ids");
        else if (field == 2) parse_packed_varints(reader, wire, result.attention_mask,
                                                  "tokens.attention_mask");
        else reader.skip(wire);
    }
}

void parse_embedding(Reader reader, WireInputs & result) {
    result.language_kind = WireInputs::LanguageKind::embedding;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) { need_wire(wire, 2, "embedding.embedding");
                          result.embedding = parse_tensor(reader.message()); }
        else if (field == 2) { need_wire(wire, 2, "embedding.attention_mask");
                               result.embedding_attention_mask = parse_tensor(reader.message()); }
        else reader.skip(wire);
    }
}

void parse_language(Reader reader, WireInputs & result) {
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) {
            need_wire(wire, 2, "language.tokens");
            if (result.language_kind != WireInputs::LanguageKind::none) {
                invalid_wire("LanguageInput.source appears more than once", "inputs.language");
            }
            parse_tokens(reader.message(), result);
        } else if (field == 2) {
            need_wire(wire, 2, "language.precomputed_embedding");
            if (result.language_kind != WireInputs::LanguageKind::none) {
                invalid_wire("LanguageInput.source appears more than once", "inputs.language");
            }
            parse_embedding(reader.message(), result);
        } else reader.skip(wire);
    }
}

WireInputs parse_inputs(Reader reader) {
    WireInputs result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        switch (field) {
            case 1: need_wire(wire, 2, "inputs.images");
                    result.images.push_back(parse_image(reader.message())); break;
            case 2: need_wire(wire, 2, "inputs.language"); parse_language(reader.message(), result); break;
            case 3: need_wire(wire, 2, "inputs.state"); result.state = parse_tensor(reader.message()); break;
            case 4: need_wire(wire, 2, "inputs.noise"); result.noise = parse_tensor(reader.message()); break;
            case 5: need_wire(wire, 2, "inputs.history");
                    result.history.push_back(parse_named_tensor(reader.message())); break;
            default: reader.skip(wire);
        }
    }
    return result;
}

WirePredictRequest parse_predict(Reader reader) {
    WirePredictRequest result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) { need_wire(wire, 2, "session_id"); result.session_id = reader.string(); }
        else if (field == 2) { need_wire(wire, 0, "request_id"); result.request_id = reader.varint(); }
        else if (field == 3) { need_wire(wire, 2, "inputs"); result.inputs = parse_inputs(reader.message()); }
        else reader.skip(wire);
    }
    return result;
}

std::vector<std::uint8_t> tensor_wire(const WireTensor & tensor) {
    Writer writer;
    writer.enum_field(1, static_cast<std::uint32_t>(tensor.dtype));
    writer.packed_u64(2, tensor.shape);
    writer.string(3, tensor.layout);
    writer.enum_field(4, static_cast<std::uint32_t>(tensor.byte_order));
    writer.bytes(5, tensor.data);
    return writer.take();
}

std::vector<std::uint8_t> image_wire(const WireImage & image) {
    Writer writer;
    writer.string(1, image.name);
    writer.enum_field(2, static_cast<std::uint32_t>(image.encoding));
    writer.u32(3, image.width); writer.u32(4, image.height); writer.u32(5, image.channels);
    writer.u64(6, image.row_stride_bytes); writer.bytes(7, image.data);
    return writer.take();
}

std::vector<std::uint8_t> named_tensor_wire(const WireNamedTensor & value) {
    Writer writer;
    writer.string(1, value.name); writer.message(2, tensor_wire(value.tensor));
    return writer.take();
}

std::vector<std::uint8_t> inputs_wire(const WireInputs & inputs) {
    Writer writer;
    for (const WireImage & image : inputs.images) writer.message(1, image_wire(image));
    Writer language;
    if (inputs.language_kind == WireInputs::LanguageKind::tokens) {
        Writer tokens;
        tokens.packed_i32(1, inputs.token_ids); tokens.packed_i32(2, inputs.attention_mask);
        language.message(1, tokens.take());
    } else if (inputs.language_kind == WireInputs::LanguageKind::embedding) {
        Writer embedding;
        embedding.message(1, tensor_wire(inputs.embedding));
        if (!inputs.embedding_attention_mask.data.empty()) {
            embedding.message(2, tensor_wire(inputs.embedding_attention_mask));
        }
        language.message(2, embedding.take());
    }
    if (inputs.language_kind != WireInputs::LanguageKind::none) {
        writer.message(2, language.take());
    }
    writer.message(3, tensor_wire(inputs.state));
    if (!inputs.noise.data.empty()) writer.message(4, tensor_wire(inputs.noise));
    for (const WireNamedTensor & value : inputs.history) {
        writer.message(5, named_tensor_wire(value));
    }
    return writer.take();
}

std::vector<std::uint8_t> predict_wire(const WirePredictRequest & request) {
    Writer writer;
    writer.string(1, request.session_id); writer.u64(2, request.request_id);
    writer.message(3, inputs_wire(request.inputs));
    return writer.take();
}

std::vector<std::int64_t> checked_shape(const WireTensor & tensor, const char * field) {
    std::vector<std::int64_t> result;
    result.reserve(tensor.shape.size());
    for (std::uint64_t dimension : tensor.shape) {
        if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw Error(ErrorCode::invalid_argument, "tensor dimension exceeds int64",
                        {{field, std::to_string(dimension)}});
        }
        result.push_back(static_cast<std::int64_t>(dimension));
    }
    return result;
}

TensorView tensor_view(const WireTensor & value, const char * field) {
    return {value.data.empty() ? nullptr : value.data.data(), value.data.size(), value.dtype,
            checked_shape(value, field), value.layout, value.byte_order};
}

Inputs runtime_inputs(WireInputs & value) {
    Inputs result;
    for (WireImage & image : value.images) {
        if (image.row_stride_bytes > std::numeric_limits<std::size_t>::max()) {
            throw Error(ErrorCode::invalid_argument, "image row stride overflows size_t",
                        {{"inputs.images.row_stride_bytes", std::to_string(image.row_stride_bytes)}});
        }
        result.images.push_back({image.name, image.encoding,
            image.data.empty() ? nullptr : image.data.data(), image.data.size(), image.width,
            image.height, image.channels, static_cast<std::size_t>(image.row_stride_bytes)});
    }
    if (value.language_kind == WireInputs::LanguageKind::tokens) {
        result.language = TokenInput{ArrayView<std::int32_t>(value.token_ids),
                                     ArrayView<std::int32_t>(value.attention_mask)};
    } else if (value.language_kind == WireInputs::LanguageKind::embedding) {
        result.language = EmbeddingInput{tensor_view(value.embedding, "inputs.language.embedding"),
            tensor_view(value.embedding_attention_mask, "inputs.language.attention_mask")};
    }
    result.state = tensor_view(value.state, "inputs.state");
    result.noise = tensor_view(value.noise, "inputs.noise");
    for (WireNamedTensor & item : value.history) {
        result.history.push_back({item.name, tensor_view(item.tensor, "inputs.history")});
    }
    return result;
}

WireTensor wire_tensor(const Tensor & tensor) {
    WireTensor result;
    result.dtype = tensor.dtype;
    for (std::int64_t dimension : tensor.shape) {
        if (dimension < 0) throw Error(ErrorCode::internal, "negative response tensor dimension");
        result.shape.push_back(static_cast<std::uint64_t>(dimension));
    }
    result.layout = tensor.layout;
    result.byte_order = tensor.byte_order;
    result.data = tensor.data;
    return result;
}

std::vector<std::uint8_t> phase_timing_wire(const PhaseTiming & timing) {
    Writer writer;
    writer.string(1, timing.name); writer.fixed64_field(2, timing.milliseconds);
    return writer.take();
}

std::vector<std::uint8_t> stats_wire(const Stats & stats) {
    Writer writer;
    writer.fixed64_field(1, stats.preprocess_milliseconds);
    writer.fixed64_field(2, stats.vae_milliseconds);
    writer.fixed64_field(3, stats.text_encoder_milliseconds);
    writer.fixed64_field(4, stats.prompt_projection_milliseconds);
    writer.fixed64_field(5, stats.prefix_milliseconds);
    writer.fixed64_field(6, stats.denoise_milliseconds);
    writer.fixed64_field(7, stats.postprocess_milliseconds);
    writer.fixed64_field(8, stats.total_milliseconds);
    writer.packed_double(9, stats.denoise_step_milliseconds);
    for (const PhaseTiming & timing : stats.additional_timings) {
        writer.message(10, phase_timing_wire(timing));
    }
    writer.boolean(11, stats.prompt_cache_hit);
    writer.boolean(12, stats.projected_prompt_cache_hit);
    writer.u64(13, stats.peak_device_memory_bytes);
    return writer.take();
}

std::vector<std::uint8_t> prediction_wire(const Prediction & prediction) {
    Writer writer;
    writer.message(1, tensor_wire(wire_tensor(prediction.action)));
    for (const NamedTensor & item : prediction.auxiliary) {
        writer.message(2, named_tensor_wire({item.name, wire_tensor(item.tensor)}));
    }
    writer.message(3, stats_wire(prediction.stats));
    return writer.take();
}

std::vector<std::uint8_t> predict_response_wire(std::uint64_t request_id,
                                                const Prediction & prediction) {
    Writer writer;
    writer.u64(1, request_id); writer.message(2, prediction_wire(prediction));
    return writer.take();
}

std::vector<std::uint8_t> version_wire(Version version) {
    Writer writer;
    writer.u32(1, version.major); writer.u32(2, version.minor); writer.u32(3, version.patch);
    return writer.take();
}

std::vector<std::uint8_t> capabilities_wire(const Capabilities & capabilities) {
    Writer writer;
    writer.boolean(1, capabilities.action);
    writer.boolean(2, capabilities.auxiliary_outputs);
    writer.boolean(3, capabilities.raw_images);
    writer.boolean(4, capabilities.token_input);
    writer.boolean(5, capabilities.precomputed_embedding);
    writer.boolean(6, capabilities.explicit_noise);
    writer.boolean(7, capabilities.batch_inference);
    writer.boolean(8, capabilities.concurrent_sessions);
    for (Backend value : capabilities.backends) writer.enum_field(9, static_cast<std::uint32_t>(value));
    for (Precision value : capabilities.precisions) writer.enum_field(10, static_cast<std::uint32_t>(value));
    writer.boolean(11, capabilities.arbitrary_token_input);
    writer.boolean(12, capabilities.fixed_token_input);
    return writer.take();
}

std::vector<std::uint8_t> runtime_component_wire(const RuntimeComponentInfo & component) {
    Writer writer;
    writer.string(1, component.name);
    writer.boolean(2, component.loaded);
    writer.u64(3, component.device_bytes);
    writer.fixed64_field(4, component.load_milliseconds);
    writer.fixed64_field(5, component.unload_milliseconds);
    return writer.take();
}

std::vector<std::uint8_t> component_wire(const ArtifactComponentInfo & component) {
    Writer writer;
    writer.string(1, component.name);
    writer.enum_field(2, static_cast<std::uint32_t>(component.source_dtype));
    writer.enum_field(3, static_cast<std::uint32_t>(component.stored_dtype));
    writer.u64(4, component.tensor_count); writer.u64(5, component.elements);
    writer.u64(6, component.source_bytes); writer.u64(7, component.stored_bytes);
    return writer.take();
}

std::vector<std::uint8_t> model_info_wire(const ModelInfo & info) {
    Writer model;
    model.string(1, info.architecture); model.string(2, info.artifact_policy);
    model.string(3, info.artifact_sha256); model.u64(4, info.artifact_bytes);
    model.enum_field(5, static_cast<std::uint32_t>(info.backend));
    model.enum_field(6, static_cast<std::uint32_t>(info.precision));
    model.message(7, capabilities_wire(info.capabilities));
    model.message(8, version_wire(runtime_version()));
    model.message(9, version_wire(protocol_version()));
    model.u32(10, artifact_schema_version()); model.u32(11, abi_version());
    for (const ArtifactComponentInfo & component : info.artifact_components) {
        model.message(12, component_wire(component));
    }
    model.enum_field(13, static_cast<std::uint32_t>(info.language_encoder_policy));
    model.boolean(14, info.text_encoder_resident);
    model.u64(15, info.resident_device_bytes);
    for (const RuntimeComponentInfo & component : info.runtime_components) {
        model.message(16, runtime_component_wire(component));
    }
    model.u64(17, info.peak_component_device_bytes);
    Writer response;
    response.message(1, model.take());
    return response.take();
}

std::vector<std::uint8_t> error_wire(ErrorCode code, const std::string & message,
                                     const std::vector<ErrorDetail> & details) {
    Writer writer;
    writer.enum_field(1, static_cast<std::uint32_t>(code)); writer.string(2, message);
    for (const ErrorDetail & detail : details) {
        Writer item;
        item.string(1, detail.field); item.string(2, detail.reason);
        writer.message(3, item.take());
    }
    return writer.take();
}

std::string parse_session_id(Reader reader, const char * message_name) {
    std::string result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) { need_wire(wire, 2, "session_id"); result = reader.string(); }
        else reader.skip(wire);
    }
    if (result.empty()) {
        throw Error(ErrorCode::invalid_argument, std::string(message_name) + " requires session_id",
                    {{"session_id", "empty"}});
    }
    return result;
}

bool parse_prefix_cache(Reader reader) {
    std::uint64_t policy = 0;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1) { need_wire(wire, 0, "prefix_cache"); policy = reader.varint(); }
        else reader.skip(wire);
    }
    if (policy > 2) {
        throw Error(ErrorCode::invalid_argument, "unknown prefix cache policy",
                    {{"prefix_cache", std::to_string(policy)}});
    }
    return policy != 2;
}

RpcResult failure(ErrorCode code, const std::string & message,
                  const std::vector<ErrorDetail> & details = {}) {
    return {code, grpc_status(code), error_wire(code, message, details), message};
}

} // namespace

std::vector<std::uint8_t> encode_predict_request(const WirePredictRequest & request) {
    return predict_wire(request);
}

WirePredictRequest decode_predict_request(const std::uint8_t * data, std::size_t size) {
    return parse_predict(Reader(data, size));
}

GrpcStatusCode grpc_status(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::ok: return GrpcStatusCode::ok;
        case ErrorCode::invalid_argument: return GrpcStatusCode::invalid_argument;
        case ErrorCode::not_found: return GrpcStatusCode::not_found;
        case ErrorCode::unsupported: return GrpcStatusCode::unimplemented;
        case ErrorCode::incompatible_artifact:
        case ErrorCode::failed_precondition: return GrpcStatusCode::failed_precondition;
        case ErrorCode::resource_exhausted: return GrpcStatusCode::resource_exhausted;
        case ErrorCode::inference_failed:
        case ErrorCode::internal: return GrpcStatusCode::internal;
    }
    return GrpcStatusCode::internal;
}

struct ProtocolService::Impl {
    explicit Impl(const ModelOptions & options) : model(model_load(options)) {}
    ~Impl() {
        for (auto & item : sessions) session_free(item.second);
        model_free(model);
    }

    RpcResult dispatch(RpcMethod method, const std::uint8_t * data, std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        Reader reader(data, size);
        switch (method) {
            case RpcMethod::get_model_info:
                while (!reader.done()) { const auto [field, wire] = reader.tag();
                                         (void) field; reader.skip(wire); }
                return {ErrorCode::ok, GrpcStatusCode::ok, model_info_wire(model_info(model)), {}};
            case RpcMethod::create_session: {
                const bool prefix_cache = parse_prefix_cache(std::move(reader));
                Session * session = session_create(model, {prefix_cache});
                std::ostringstream id;
                id << "session-" << next_session++;
                sessions.emplace(id.str(), session);
                Writer response; response.string(1, id.str());
                return {ErrorCode::ok, GrpcStatusCode::ok, response.take(), {}};
            }
            case RpcMethod::predict: {
                WirePredictRequest request = parse_predict(std::move(reader));
                if (request.session_id.empty()) {
                    throw Error(ErrorCode::invalid_argument, "Predict requires session_id",
                                {{"session_id", "empty"}});
                }
                const auto found = sessions.find(request.session_id);
                if (found == sessions.end()) {
                    throw Error(ErrorCode::not_found, "session does not exist",
                                {{"session_id", request.session_id}});
                }
                Inputs inputs = runtime_inputs(request.inputs);
                Prediction prediction = wam::predict(found->second, inputs);
                return {ErrorCode::ok, GrpcStatusCode::ok,
                        predict_response_wire(request.request_id, prediction), {}};
            }
            case RpcMethod::reset_session: {
                const std::string id = parse_session_id(std::move(reader), "ResetSession");
                const auto found = sessions.find(id);
                if (found == sessions.end()) {
                    throw Error(ErrorCode::not_found, "session does not exist", {{"session_id", id}});
                }
                const Status status = session_reset(found->second);
                if (!status) return failure(status.code, status.message, status.details);
                return {ErrorCode::ok, GrpcStatusCode::ok, {}, {}};
            }
            case RpcMethod::close_session: {
                const std::string id = parse_session_id(std::move(reader), "CloseSession");
                const auto found = sessions.find(id);
                if (found == sessions.end()) {
                    throw Error(ErrorCode::not_found, "session does not exist", {{"session_id", id}});
                }
                session_free(found->second);
                sessions.erase(found);
                return {ErrorCode::ok, GrpcStatusCode::ok, {}, {}};
            }
        }
        throw Error(ErrorCode::invalid_argument, "unknown RPC method",
                    {{"method", std::to_string(static_cast<std::uint32_t>(method))}});
    }

    Model * model = nullptr;
    std::map<std::string, Session *> sessions;
    std::uint64_t next_session = 1;
    std::mutex mutex;
};

ProtocolService::ProtocolService(const ModelOptions & options)
    : impl_(std::make_unique<Impl>(options)) {}

ProtocolService::~ProtocolService() = default;

RpcResult ProtocolService::call(RpcMethod method, const std::uint8_t * request,
                                std::size_t size) noexcept {
    try {
        return impl_->dispatch(method, request, size);
    } catch (const Error & error) {
        return failure(error.code(), error.what(), error.details());
    } catch (const std::exception & error) {
        return failure(ErrorCode::internal, error.what());
    } catch (...) {
        return failure(ErrorCode::internal, "unknown RPC adapter failure");
    }
}

} // namespace wam::serving
