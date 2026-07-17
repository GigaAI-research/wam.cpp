#include "gguf_reader.h"

#include <filesystem>
#include <cstring>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

namespace wam::internal {
namespace {

[[noreturn]] void artifact_error(const std::string & message,
                                 const std::string & field,
                                 const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

const char * type_name(gguf_type type) {
    const char * name = gguf_type_name(type);
    return name == nullptr ? "unknown" : name;
}

} // namespace

std::shared_ptr<GgufReader> GgufReader::open(const std::string & path) {
    auto reader = std::shared_ptr<GgufReader>(new GgufReader(path));
    reader->initialize();
    return reader;
}

GgufReader::GgufReader(std::string path) : path_(std::move(path)) {}

GgufReader::~GgufReader() {
    if (file_ != nullptr) std::fclose(file_);
    if (context_ != nullptr) gguf_free(context_);
    if (metadata_context_ != nullptr) ggml_free(metadata_context_);
}

void GgufReader::initialize() {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path_, error)) {
        throw Error(ErrorCode::not_found, "model artifact is not a regular file",
                    {{"artifact_path", path_}});
    }
    file_size_ = std::filesystem::file_size(path_, error);
    if (error) {
        throw Error(ErrorCode::not_found, "cannot determine artifact size",
                    {{"artifact_path", path_}});
    }

    gguf_init_params parameters{};
    parameters.no_alloc = true;
    parameters.ctx = &metadata_context_;
    context_ = gguf_init_from_file(path_.c_str(), parameters);
    if (context_ == nullptr || metadata_context_ == nullptr) {
        artifact_error("cannot parse GGUF metadata", "artifact_path", path_);
    }
    data_offset_ = gguf_get_data_offset(context_);
    file_ = std::fopen(path_.c_str(), "rb");
    if (file_ == nullptr) {
        throw Error(ErrorCode::not_found, "cannot open model artifact payload",
                    {{"artifact_path", path_}});
    }

    const std::int64_t count = gguf_get_n_tensors(context_);
    if (count < 0) artifact_error("invalid GGUF tensor count", "tensors", "negative count");
    tensors_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        const char * raw_name = gguf_get_tensor_name(context_, index);
        if (raw_name == nullptr || *raw_name == '\0') {
            artifact_error("GGUF tensor has no name", "tensors", std::to_string(index));
        }
        const ggml_tensor * tensor = ggml_get_tensor(metadata_context_, raw_name);
        if (tensor == nullptr) {
            artifact_error("GGUF tensor descriptor is unavailable", raw_name, "missing descriptor");
        }

        GgufTensorInfo info;
        info.name = raw_name;
        info.type = tensor->type;
        const int dimensions = ggml_n_dims(tensor);
        info.shape.assign(tensor->ne, tensor->ne + dimensions);
        info.elements = static_cast<std::uint64_t>(ggml_nelements(tensor));
        info.bytes = static_cast<std::uint64_t>(ggml_nbytes(tensor));
        const std::uint64_t relative = gguf_get_tensor_offset(context_, index);
        if (relative > std::numeric_limits<std::uint64_t>::max() - data_offset_) {
            artifact_error("GGUF tensor offset overflows", info.name, "invalid offset");
        }
        info.file_offset = data_offset_ + relative;
        if (info.file_offset > file_size_ || info.bytes > file_size_ - info.file_offset) {
            artifact_error("GGUF tensor payload is outside the artifact", info.name,
                           "truncated payload");
        }
        tensors_.push_back(std::move(info));
    }
}

bool GgufReader::has(const std::string & key) const noexcept {
    return gguf_find_key(context_, key.c_str()) >= 0;
}

std::int64_t GgufReader::require_key(const std::string & key, gguf_type type) const {
    const std::int64_t index = gguf_find_key(context_, key.c_str());
    if (index < 0) artifact_error("GGUF metadata key is required", key, "missing");
    const gguf_type actual = gguf_get_kv_type(context_, index);
    if (actual != type) {
        artifact_error("GGUF metadata has the wrong type", key,
                       std::string("expected ") + type_name(type) + ", got " + type_name(actual));
    }
    return index;
}

std::string GgufReader::require_string(const std::string & key) const {
    const char * value = gguf_get_val_str(context_, require_key(key, GGUF_TYPE_STRING));
    return value == nullptr ? std::string() : std::string(value);
}

std::string GgufReader::optional_string(const std::string & key,
                                        std::string fallback) const {
    if (!has(key)) return fallback;
    return require_string(key);
}

std::uint32_t GgufReader::require_u32(const std::string & key) const {
    return gguf_get_val_u32(context_, require_key(key, GGUF_TYPE_UINT32));
}

float GgufReader::require_f32(const std::string & key) const {
    return gguf_get_val_f32(context_, require_key(key, GGUF_TYPE_FLOAT32));
}

std::uint64_t GgufReader::require_u64(const std::string & key) const {
    return gguf_get_val_u64(context_, require_key(key, GGUF_TYPE_UINT64));
}

std::vector<float> GgufReader::require_f32_array(const std::string & key) const {
    const std::int64_t index = require_key(key, GGUF_TYPE_ARRAY);
    const gguf_type type = gguf_get_arr_type(context_, index);
    const std::size_t count = gguf_get_arr_n(context_, index);
    std::vector<float> result(count);
    if (type == GGUF_TYPE_FLOAT32) {
        std::memcpy(result.data(), gguf_get_arr_data(context_, index),
                    count * sizeof(float));
        return result;
    }
    if (type == GGUF_TYPE_FLOAT64) {
        const auto * source = static_cast<const double *>(gguf_get_arr_data(context_, index));
        for (std::size_t i = 0; i < count; ++i) result[i] = static_cast<float>(source[i]);
        return result;
    }
    artifact_error("GGUF metadata array has the wrong element type", key,
                   "expected F32 or F64");
}

std::vector<float> GgufReader::optional_f32_array(const std::string & key) const {
    return has(key) ? require_f32_array(key) : std::vector<float>{};
}

std::size_t GgufReader::tensor_count() const noexcept { return tensors_.size(); }

const GgufTensorInfo * GgufReader::find_tensor(const std::string & name) const noexcept {
    for (const GgufTensorInfo & tensor : tensors_) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

const GgufTensorInfo & GgufReader::require_tensor(const std::string & name) const {
    const GgufTensorInfo * tensor = find_tensor(name);
    if (tensor == nullptr) artifact_error("required GGUF tensor is missing", name, "missing");
    return *tensor;
}

const ggml_tensor * GgufReader::descriptor(const std::string & name) const noexcept {
    return ggml_get_tensor(metadata_context_, name.c_str());
}

bool GgufReader::read_tensor(const std::string & name, void * destination,
                             std::size_t bytes) const {
    const GgufTensorInfo * tensor = find_tensor(name);
    if (tensor == nullptr || destination == nullptr || bytes != tensor->bytes ||
        file_ == nullptr || tensor->file_offset >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        return false;
    }
    if (fseeko(file_, static_cast<off_t>(tensor->file_offset), SEEK_SET) != 0) return false;
    return std::fread(destination, 1, bytes, file_) == bytes;
}

std::vector<float> GgufReader::read_f32_tensor(const std::string & name) const {
    const GgufTensorInfo & info = require_tensor(name);
    std::vector<float> result(static_cast<std::size_t>(info.elements));
    if (info.type == GGML_TYPE_F32) {
        if (!read_tensor(name, result.data(), static_cast<std::size_t>(info.bytes))) return {};
        return result;
    }
    if (info.type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> source(static_cast<std::size_t>(info.elements));
        if (!read_tensor(name, source.data(), static_cast<std::size_t>(info.bytes))) return {};
        ggml_bf16_to_fp32_row(source.data(), result.data(),
                              static_cast<std::int64_t>(source.size()));
        return result;
    }
    artifact_error("GGUF tensor cannot be converted to F32", name, "expected F32 or BF16");
}

void GgufReader::require_shape(const std::string & name,
                               const std::vector<std::int64_t> & shape) const {
    const GgufTensorInfo & tensor = require_tensor(name);
    auto normalized = [](std::vector<std::int64_t> value) {
        while (value.size() > 1 && value.back() == 1) value.pop_back();
        return value;
    };
    if (normalized(tensor.shape) == normalized(shape)) return;
    std::ostringstream reason;
    reason << "expected [";
    for (std::size_t i = 0; i < shape.size(); ++i) reason << (i ? "," : "") << shape[i];
    reason << "], got [";
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        reason << (i ? "," : "") << tensor.shape[i];
    }
    reason << ']';
    artifact_error("GGUF tensor has the wrong shape", name, reason.str());
}

} // namespace wam::internal
