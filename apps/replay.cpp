#include "replay.h"

#include "json.hpp"

extern "C" {
#include "sha256.h"
}

#if WAM_APP_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

namespace wam::apps {
namespace {

using json = nlohmann::json;

std::vector<std::uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    const std::streamsize size = input.tellg();
    if (size < 0) throw std::runtime_error("cannot size " + path.string());
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    input.seekg(0);
    if (size && !input.read(reinterpret_cast<char *>(result.data()), size)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return result;
}

ImageEncoding image_encoding(const std::string & value,
                             const std::filesystem::path & path = {}) {
    std::string name = value;
    if (name.empty()) name = path.extension().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (name == "png" || name == ".png") return ImageEncoding::png;
    if (name == "jpeg" || name == "jpg" || name == ".jpeg" || name == ".jpg") {
        return ImageEncoding::jpeg;
    }
    if (name == "rgb_u8" || name == "rgb") return ImageEncoding::rgb_u8;
    throw std::runtime_error("unsupported image encoding: " + name);
}

DType tensor_dtype(const std::string & value) {
    if (value == "float32" || value == "f32") return DType::f32;
    if (value == "bfloat16" || value == "bf16") return DType::bf16;
    if (value == "int32" || value == "i32") return DType::i32;
    if (value == "uint8" || value == "u8") return DType::u8;
    throw std::runtime_error("unsupported tensor dtype: " + value);
}

OwnedTensor json_tensor(const json & value, const std::filesystem::path & base,
                        std::vector<std::int64_t> default_shape,
                        const std::string & default_layout) {
    OwnedTensor result;
    result.dtype = tensor_dtype(value.value("dtype", "float32"));
    result.byte_order = ByteOrder::little;
    result.layout = value.value("layout", default_layout);
    result.shape = value.contains("shape")
        ? value.at("shape").get<std::vector<std::int64_t>>() : std::move(default_shape);
    if (value.contains("path")) {
        result.data = read_bytes(base / value.at("path").get<std::string>());
    } else if (value.contains("values")) {
        if (result.dtype != DType::f32) {
            throw std::runtime_error("inline tensor values currently require F32");
        }
        const std::vector<float> numbers = value.at("values").get<std::vector<float>>();
        result.data.resize(numbers.size() * sizeof(float));
        std::memcpy(result.data.data(), numbers.data(), result.data.size());
    } else {
        throw std::runtime_error("tensor requires path or values");
    }
    return result;
}

class WireReader {
public:
    WireReader(const std::uint8_t * data, std::size_t size) : current_(data), end_(data + size) {}

    bool done() const { return current_ == end_; }
    std::uint64_t varint() {
        std::uint64_t result = 0;
        for (unsigned shift = 0; shift < 64; shift += 7) {
            if (current_ == end_) throw std::runtime_error("truncated protobuf varint");
            const std::uint8_t byte = *current_++;
            result |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return result;
        }
        throw std::runtime_error("invalid protobuf varint");
    }
    std::pair<unsigned, unsigned> tag() {
        const std::uint64_t value = varint();
        return {static_cast<unsigned>(value >> 3), static_cast<unsigned>(value & 7)};
    }
    WireReader message() {
        const std::size_t size = static_cast<std::size_t>(varint());
        if (size > static_cast<std::size_t>(end_ - current_)) {
            throw std::runtime_error("truncated protobuf message");
        }
        WireReader result(current_, size);
        current_ += size;
        return result;
    }
    std::vector<std::uint8_t> bytes() {
        WireReader value = message();
        return {value.current_, value.end_};
    }
    std::string string() {
        const std::vector<std::uint8_t> value = bytes();
        return {reinterpret_cast<const char *>(value.data()), value.size()};
    }
    void skip(unsigned wire) {
        if (wire == 0) { (void) varint(); return; }
        if (wire == 1) { advance(8); return; }
        if (wire == 2) { WireReader ignored = message(); (void) ignored; return; }
        if (wire == 5) { advance(4); return; }
        throw std::runtime_error("unsupported protobuf wire type");
    }

private:
    void advance(std::size_t size) {
        if (size > static_cast<std::size_t>(end_ - current_)) {
            throw std::runtime_error("truncated protobuf field");
        }
        current_ += size;
    }
    const std::uint8_t * current_;
    const std::uint8_t * end_;
};

OwnedTensor protobuf_tensor(WireReader reader, const std::string & layout) {
    OwnedTensor result;
    result.layout = layout;
    result.byte_order = ByteOrder::little;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1 && wire == 2) result.dtype = tensor_dtype(reader.string());
        else if (field == 2 && wire == 0) {
            result.shape.push_back(static_cast<std::int64_t>(reader.varint()));
        } else if (field == 2 && wire == 2) {
            WireReader packed = reader.message();
            while (!packed.done()) {
                result.shape.push_back(static_cast<std::int64_t>(packed.varint()));
            }
        } else if (field == 3 && wire == 2) {
            if (reader.string() != "little") throw std::runtime_error("tensor is not little-endian");
        } else if (field == 4 && wire == 2) result.data = reader.bytes();
        else reader.skip(wire);
    }
    if (result.dtype == DType::unknown || result.shape.empty() || result.data.empty()) {
        throw std::runtime_error("incomplete protobuf tensor");
    }
    return result;
}

OwnedImage protobuf_image(WireReader reader) {
    OwnedImage result;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1 && wire == 2) result.name = reader.string();
        else if (field == 2 && wire == 2) result.encoding = image_encoding(reader.string());
        else if (field == 3 && wire == 0) result.width = static_cast<std::uint32_t>(reader.varint());
        else if (field == 4 && wire == 0) result.height = static_cast<std::uint32_t>(reader.varint());
        else if (field == 5 && wire == 0) result.channels = static_cast<std::uint32_t>(reader.varint());
        else if (field == 6 && wire == 2) result.data = reader.bytes();
        else reader.skip(wire);
    }
    if (result.name.empty() || result.encoding == ImageEncoding::unknown || result.data.empty()) {
        throw std::runtime_error("incomplete protobuf image");
    }
    return result;
}

ReplayRequest load_protobuf(const std::filesystem::path & path) {
    const std::vector<std::uint8_t> bytes = read_bytes(path);
    WireReader reader(bytes.data(), bytes.size());
    ReplayRequest result;
    result.format = "wam-replay-protobuf-v1";
    std::vector<std::string> camera_order;
    std::string scheduler;
    std::uint64_t steps = 0;
    while (!reader.done()) {
        const auto [field, wire] = reader.tag();
        if (field == 1 && wire == 0) {
            if (reader.varint() != 1) throw std::runtime_error("unsupported replay version");
        } else if (field == 2 && wire == 2) result.images.push_back(protobuf_image(reader.message()));
        else if (field == 4 && wire == 2) result.prompt_embedding = protobuf_tensor(reader.message(), "T,D");
        else if (field == 5 && wire == 2) result.state = protobuf_tensor(reader.message(), "D");
        else if (field == 6 && wire == 2) result.noise = protobuf_tensor(reader.message(), "B,T,A");
        else if (field == 7 && wire == 2) camera_order.push_back(reader.string());
        else if (field == 8 && wire == 2) scheduler = reader.string();
        else if (field == 9 && wire == 0) steps = reader.varint();
        else reader.skip(wire);
    }
    const std::vector<std::string> expected = {
        kImageHigh, kImageLeftWrist, kImageRightWrist,
    };
    if (camera_order != expected || scheduler != "flow-match-euler" || steps != 10) {
        throw std::runtime_error("replay semantics do not match GWP-0.5");
    }
    return result;
}

ReplayRequest load_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    json root;
    input >> root;
    ReplayRequest result;
    result.format = root.value("format", "wam-replay-json-v1");
    const std::filesystem::path base = path.parent_path();
    for (const json & item : root.at("images")) {
        OwnedImage image;
        image.name = item.value("logical_name", item.value("name", ""));
        const std::filesystem::path image_path = base / item.at("path").get<std::string>();
        image.encoding = image_encoding(item.value("encoding", ""), image_path);
        image.data = read_bytes(image_path);
        if (item.contains("shape")) {
            const auto shape = item.at("shape").get<std::vector<std::uint32_t>>();
            if (shape.size() == 3) {
                image.height = shape[0]; image.width = shape[1]; image.channels = shape[2];
            }
        }
        result.images.push_back(std::move(image));
    }
    const json & language = root.at("language");
    if (language.contains("precomputed_embedding")) {
        result.prompt_embedding = json_tensor(
            language.at("precomputed_embedding"), base, {64, 4096}, "T,D");
    } else if (language.contains("tokens")) {
        const json & tokens = language.at("tokens");
        result.token_ids = tokens.at("token_ids").get<std::vector<std::int32_t>>();
        result.attention_mask = tokens.value(
            "attention_mask", std::vector<std::int32_t>(result.token_ids.size(), 1));
    } else {
        throw std::runtime_error("language requires precomputed_embedding or tokens");
    }
    result.state = json_tensor(root.at("state"), base, {14}, "D");
    result.noise = json_tensor(root.at("noise"), base, {1, 48, 32}, "B,T,A");
    return result;
}

} // namespace

TensorView OwnedTensor::view() const {
    return {data.data(), data.size(), dtype, shape, layout, byte_order};
}

Inputs ReplayRequest::inputs() const {
    Inputs result;
    for (const OwnedImage & image : images) {
        result.images.push_back({image.name, image.encoding, image.data.data(), image.data.size(),
                                 image.width, image.height, image.channels,
                                 image.row_stride_bytes});
    }
    if (!prompt_embedding.data.empty()) {
        result.language = EmbeddingInput{prompt_embedding.view(), {}};
    } else {
        result.language = TokenInput{ArrayView<std::int32_t>(token_ids),
                                     ArrayView<std::int32_t>(attention_mask)};
    }
    result.state = state.view();
    result.noise = noise.view();
    return result;
}

ReplayRequest load_request(const std::filesystem::path & path) {
    if (path.extension() == ".pb") return load_protobuf(path);
    if (path.extension() == ".json") return load_json(path);
    throw std::runtime_error("request must end in .pb or .json");
}

Precision parse_precision(const std::string & value) {
    if (value == "f32" || value == "reference") return Precision::f32_reference;
    if (value == "bf16" || value == "latency") return Precision::bf16_latency;
    throw std::runtime_error("precision must be f32/reference or bf16/latency");
}

std::string precision_name(Precision precision) {
    return precision == Precision::bf16_latency ? "bf16_latency" : "f32_reference";
}

LanguageEncoderPolicy parse_language_policy(const std::string & value) {
    if (value == "resident") return LanguageEncoderPolicy::resident;
    if (value == "fixed") return LanguageEncoderPolicy::fixed;
    if (value == "external_embedding" || value == "external") {
        return LanguageEncoderPolicy::external_embedding;
    }
    throw std::runtime_error(
        "language policy must be resident, fixed, or external_embedding");
}

std::string language_policy_name(LanguageEncoderPolicy policy) {
    switch (policy) {
        case LanguageEncoderPolicy::resident: return "resident";
        case LanguageEncoderPolicy::fixed: return "fixed";
        case LanguageEncoderPolicy::external_embedding: return "external_embedding";
    }
    return "unknown";
}

std::string sha256_hex(const void * data, std::size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : digest) output << std::setw(2) << +value;
    return output.str();
}

std::vector<float> action_values(const Tensor & action) {
    if (action.dtype != DType::f32 || action.byte_order != ByteOrder::little ||
        action.data.size() % sizeof(float) != 0) {
        throw std::runtime_error("action is not little-endian F32");
    }
    std::vector<float> result(action.data.size() / sizeof(float));
    std::memcpy(result.data(), action.data.data(), action.data.size());
    return result;
}

Comparison compare_action(const Tensor & action, const std::filesystem::path & expected) {
    const std::vector<float> actual = action_values(action);
    const std::vector<std::uint8_t> expected_bytes = read_bytes(expected);
    if (expected_bytes.size() != action.data.size()) {
        throw std::runtime_error("expected action byte size differs from actual action");
    }
    std::vector<float> reference(actual.size());
    std::memcpy(reference.data(), expected_bytes.data(), expected_bytes.size());
    Comparison result;
    result.present = true;
    result.finite = true;
    result.elements = actual.size();
    double total = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) result.finite = false;
        const double error = std::fabs(static_cast<double>(actual[index]) - reference[index]);
        total += error;
        result.max_abs = std::max(result.max_abs, error);
    }
    result.mean_abs = total / actual.size();
    result.expected_sha256 = sha256_hex(expected_bytes.data(), expected_bytes.size());
    return result;
}

json stats_json(const Stats & stats) {
    json additional = json::array();
    for (const PhaseTiming & timing : stats.additional_timings) {
        additional.push_back({{"name", timing.name}, {"milliseconds", timing.milliseconds}});
    }
    return {{"preprocess_ms", stats.preprocess_milliseconds},
            {"vae_ms", stats.vae_milliseconds},
            {"text_encoder_ms", stats.text_encoder_milliseconds},
            {"prompt_projection_ms", stats.prompt_projection_milliseconds},
            {"prefix_ms", stats.prefix_milliseconds},
            {"denoise_ms", stats.denoise_milliseconds},
            {"postprocess_ms", stats.postprocess_milliseconds},
            {"total_ms", stats.total_milliseconds},
            {"denoise_steps_ms", stats.denoise_step_milliseconds},
            {"additional", additional},
            {"prompt_cache_hit", stats.prompt_cache_hit},
            {"projected_prompt_cache_hit", stats.projected_prompt_cache_hit},
            {"peak_device_memory_bytes", stats.peak_device_memory_bytes}};
}

json model_json(const ModelInfo & info) {
    json components = json::array();
    for (const ArtifactComponentInfo & component : info.artifact_components) {
        components.push_back({{"name", component.name},
                              {"tensor_count", component.tensor_count},
                              {"stored_bytes", component.stored_bytes}});
    }
    json runtime_components = json::array();
    for (const RuntimeComponentInfo & component : info.runtime_components) {
        runtime_components.push_back({{"name", component.name}, {"loaded", component.loaded},
                                      {"device_bytes", component.device_bytes},
                                      {"load_ms", component.load_milliseconds},
                                      {"unload_ms", component.unload_milliseconds}});
    }
    return {{"architecture", info.architecture}, {"artifact_path", info.artifact_path},
            {"artifact_policy", info.artifact_policy},
            {"artifact_sha256", info.artifact_sha256},
            {"artifact_bytes", info.artifact_bytes},
            {"precision", precision_name(info.precision)},
            {"language_encoder_policy", language_policy_name(info.language_encoder_policy)},
            {"text_encoder_resident", info.text_encoder_resident},
            {"resident_device_bytes", info.resident_device_bytes},
            {"peak_component_device_bytes", info.peak_component_device_bytes},
            {"components", components}, {"runtime_components", runtime_components}};
}

json device_json(std::int32_t device_index) {
#if WAM_APP_CUDA
    cudaDeviceProp properties{};
    int driver = 0, runtime = 0;
    if (cudaGetDeviceProperties(&properties, device_index) == cudaSuccess) {
        (void) cudaDriverGetVersion(&driver);
        (void) cudaRuntimeGetVersion(&runtime);
        return {{"backend", "cuda"}, {"index", device_index}, {"name", properties.name},
                {"compute_capability", std::to_string(properties.major) + "." +
                                           std::to_string(properties.minor)},
                {"total_memory_bytes", properties.totalGlobalMem},
                {"driver_version", driver}, {"runtime_version", runtime}};
    }
#endif
    return {{"backend", "unavailable"}, {"index", device_index}};
}

json comparison_json(const Comparison & comparison) {
    if (!comparison.present) return nullptr;
    return {{"elements", comparison.elements}, {"finite", comparison.finite},
            {"mean_abs", comparison.mean_abs}, {"max_abs", comparison.max_abs},
            {"expected_sha256", comparison.expected_sha256}};
}

json action_json(const Tensor & action, bool include_values) {
    const std::vector<float> values = action_values(action);
    bool finite = true;
    for (float value : values) finite = finite && std::isfinite(value);
    json result = {{"dtype", "f32"}, {"byte_order", "little"},
                   {"layout", action.layout}, {"shape", action.shape},
                   {"elements", values.size()}, {"finite", finite},
                   {"sha256", sha256_hex(action.data.data(), action.data.size())}};
    if (include_values) result["values"] = values;
    return result;
}

void write_action(const Tensor & action, const std::filesystem::path & path,
                  const std::string & format) {
    if (format == "f32") {
        std::ofstream output(path, std::ios::binary);
        if (!output || !output.write(reinterpret_cast<const char *>(action.data.data()),
                                     static_cast<std::streamsize>(action.data.size()))) {
            throw std::runtime_error("cannot write action " + path.string());
        }
        return;
    }
    if (format != "json") throw std::runtime_error("action format must be f32 or json");
    write_json(action_json(action, true), path.string());
}

void write_json(const json & value, const std::string & path) {
    if (path.empty() || path == "-") {
        std::cout << value.dump(2) << '\n';
        return;
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open " + path);
    output << value.dump(2) << '\n';
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

json sample_summary(const std::vector<double> & values) {
    const double mean = values.empty() ? 0.0 :
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return {{"mean", mean}, {"p50", percentile(values, 0.5)},
            {"p95", percentile(values, 0.95)}, {"samples", values}};
}

long peak_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
#if defined(__APPLE__)
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

} // namespace wam::apps
