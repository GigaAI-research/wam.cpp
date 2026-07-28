#include "artifact/gguf_reader.h"

#include "wam/error.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
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

DType to_dtype(ggml_type type) noexcept {
    switch (type) {
        case GGML_TYPE_F32:
            return DType::f32;
        case GGML_TYPE_BF16:
            return DType::bf16;
        case GGML_TYPE_I32:
            return DType::i32;
        default:
            return DType::unknown;
    }
}

std::vector<std::int64_t> normalized_shape(
    std::vector<std::int64_t> shape) {
    while (shape.size() > 1 && shape.back() == 1) {
        shape.pop_back();
    }
    return shape;
}

std::string shape_string(const std::vector<std::int64_t> & shape) {
    std::ostringstream result;
    result << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            result << ',';
        }
        result << shape[index];
    }
    result << ']';
    return result.str();
}

} // namespace

struct GgufReader::Impl {
    ~Impl() {
        if (file != nullptr) {
            std::fclose(file);
        }
        if (context != nullptr) {
            gguf_free(context);
        }
        if (metadata_context != nullptr) {
            ggml_free(metadata_context);
        }
    }

    gguf_context * context = nullptr;
    ggml_context * metadata_context = nullptr;
    std::FILE * file = nullptr;
};

std::shared_ptr<GgufReader> GgufReader::open(const std::string & path) {
    auto reader = std::shared_ptr<GgufReader>(new GgufReader(path));
    reader->initialize();
    return reader;
}

GgufReader::GgufReader(std::string path)
    : path_(std::move(path)), impl_(std::make_unique<Impl>()) {}

GgufReader::~GgufReader() = default;

void GgufReader::initialize() {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path_, error)) {
        throw Error(ErrorCode::not_found,
                    "model artifact is not a regular file",
                    {{"artifact_path", path_}});
    }
    file_size_ = std::filesystem::file_size(path_, error);
    if (error) {
        throw Error(ErrorCode::not_found,
                    "cannot determine model artifact size",
                    {{"artifact_path", path_}});
    }

    gguf_init_params parameters{};
    parameters.no_alloc = true;
    parameters.ctx = &impl_->metadata_context;
    impl_->context = gguf_init_from_file(path_.c_str(), parameters);
    if (impl_->context == nullptr || impl_->metadata_context == nullptr) {
        artifact_error("cannot parse GGUF metadata", "artifact_path", path_);
    }

    data_offset_ = gguf_get_data_offset(impl_->context);
    impl_->file = std::fopen(path_.c_str(), "rb");
    if (impl_->file == nullptr) {
        throw Error(ErrorCode::not_found,
                    "cannot open model artifact payload",
                    {{"artifact_path", path_}});
    }

    const std::int64_t count = gguf_get_n_tensors(impl_->context);
    if (count < 0) {
        artifact_error("invalid GGUF tensor count", "tensors", "negative");
    }
    tensors_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        const char * raw_name = gguf_get_tensor_name(impl_->context, index);
        if (raw_name == nullptr || *raw_name == '\0') {
            artifact_error("GGUF tensor has no name", "tensors",
                           std::to_string(index));
        }
        const ggml_tensor * tensor =
            ggml_get_tensor(impl_->metadata_context, raw_name);
        if (tensor == nullptr) {
            artifact_error("GGUF tensor descriptor is unavailable", raw_name,
                           "missing descriptor");
        }

        const std::int64_t elements = ggml_nelements(tensor);
        if (elements < 0) {
            artifact_error("GGUF tensor has a negative element count", raw_name,
                           "invalid descriptor");
        }

        GgufTensorInfo info;
        info.name = raw_name;
        info.dtype = to_dtype(tensor->type);
        const int dimensions = ggml_n_dims(tensor);
        info.shape.assign(tensor->ne, tensor->ne + dimensions);
        info.elements = static_cast<std::uint64_t>(elements);
        info.bytes = static_cast<std::uint64_t>(ggml_nbytes(tensor));

        const std::uint64_t relative =
            gguf_get_tensor_offset(impl_->context, index);
        if (relative > std::numeric_limits<std::uint64_t>::max() -
                           data_offset_) {
            artifact_error("GGUF tensor offset overflows", info.name,
                           "invalid offset");
        }
        info.file_offset = data_offset_ + relative;
        if (info.file_offset > file_size_ ||
            info.bytes > file_size_ - info.file_offset) {
            artifact_error("GGUF tensor payload is outside the artifact",
                           info.name, "truncated payload");
        }
        tensors_.push_back(std::move(info));
    }
}

const std::string & GgufReader::path() const noexcept {
    return path_;
}

std::uint64_t GgufReader::file_size() const noexcept {
    return file_size_;
}

std::uint64_t GgufReader::data_offset() const noexcept {
    return data_offset_;
}

bool GgufReader::has(const std::string & key) const noexcept {
    return gguf_find_key(impl_->context, key.c_str()) >= 0;
}

namespace {

std::int64_t require_key(const gguf_context * context,
                         const std::string & key, gguf_type expected) {
    const std::int64_t index = gguf_find_key(context, key.c_str());
    if (index < 0) {
        artifact_error("GGUF metadata key is required", key, "missing");
    }
    const gguf_type actual = gguf_get_kv_type(context, index);
    if (actual != expected) {
        artifact_error("GGUF metadata has the wrong type", key,
                       std::string("expected ") + type_name(expected) +
                           ", got " + type_name(actual));
    }
    return index;
}

std::int64_t require_array(const gguf_context * context,
                           const std::string & key, gguf_type element_type) {
    const std::int64_t index = require_key(context, key, GGUF_TYPE_ARRAY);
    const gguf_type actual = gguf_get_arr_type(context, index);
    if (actual != element_type) {
        artifact_error("GGUF metadata array has the wrong element type", key,
                       std::string("expected ") + type_name(element_type) +
                           ", got " + type_name(actual));
    }
    return index;
}

} // namespace

std::string GgufReader::require_string(const std::string & key) const {
    const char * value = gguf_get_val_str(
        impl_->context, require_key(impl_->context, key, GGUF_TYPE_STRING));
    return value == nullptr ? std::string{} : std::string(value);
}

std::string GgufReader::optional_string(const std::string & key,
                                        std::string fallback) const {
    return has(key) ? require_string(key) : std::move(fallback);
}

std::uint32_t GgufReader::require_u32(const std::string & key) const {
    return gguf_get_val_u32(
        impl_->context, require_key(impl_->context, key, GGUF_TYPE_UINT32));
}

std::uint64_t GgufReader::require_u64(const std::string & key) const {
    return gguf_get_val_u64(
        impl_->context,
        require_key(impl_->context, key, GGUF_TYPE_UINT64));
}

float GgufReader::require_f32(const std::string & key) const {
    return gguf_get_val_f32(
        impl_->context, require_key(impl_->context, key, GGUF_TYPE_FLOAT32));
}

bool GgufReader::require_bool(const std::string & key) const {
    return gguf_get_val_bool(
        impl_->context, require_key(impl_->context, key, GGUF_TYPE_BOOL));
}

std::vector<float> GgufReader::require_f32_array(
    const std::string & key) const {
    const std::int64_t index =
        require_array(impl_->context, key, GGUF_TYPE_FLOAT32);
    const std::size_t count = gguf_get_arr_n(impl_->context, index);
    if (count == 0) {
        return {};
    }
    const auto * data = static_cast<const float *>(
        gguf_get_arr_data(impl_->context, index));
    return std::vector<float>(data, data + count);
}

std::vector<float> GgufReader::optional_f32_array(
    const std::string & key) const {
    return has(key) ? require_f32_array(key) : std::vector<float>{};
}

std::vector<std::string> GgufReader::require_string_array(
    const std::string & key) const {
    const std::int64_t index =
        require_array(impl_->context, key, GGUF_TYPE_STRING);
    const std::size_t count = gguf_get_arr_n(impl_->context, index);
    std::vector<std::string> values;
    values.reserve(count);
    for (std::size_t item = 0; item < count; ++item) {
        const char * value = gguf_get_arr_str(impl_->context, index, item);
        values.emplace_back(value == nullptr ? "" : value);
    }
    return values;
}

std::vector<std::uint32_t> GgufReader::require_u32_array(
    const std::string & key) const {
    const std::int64_t index =
        require_array(impl_->context, key, GGUF_TYPE_UINT32);
    const std::size_t count = gguf_get_arr_n(impl_->context, index);
    if (count == 0) {
        return {};
    }
    const auto * data = static_cast<const std::uint32_t *>(
        gguf_get_arr_data(impl_->context, index));
    return std::vector<std::uint32_t>(data, data + count);
}

std::vector<std::int32_t> GgufReader::require_i32_array(
    const std::string & key) const {
    const std::int64_t index =
        require_array(impl_->context, key, GGUF_TYPE_INT32);
    const std::size_t count = gguf_get_arr_n(impl_->context, index);
    if (count == 0) {
        return {};
    }
    const auto * data = static_cast<const std::int32_t *>(
        gguf_get_arr_data(impl_->context, index));
    return std::vector<std::int32_t>(data, data + count);
}

std::size_t GgufReader::tensor_count() const noexcept {
    return tensors_.size();
}

const GgufTensorInfo * GgufReader::find_tensor(
    const std::string & name) const noexcept {
    for (const GgufTensorInfo & tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

const GgufTensorInfo & GgufReader::require_tensor(
    const std::string & name) const {
    const GgufTensorInfo * tensor = find_tensor(name);
    if (tensor == nullptr) {
        artifact_error("required GGUF tensor is missing", name, "missing");
    }
    return *tensor;
}

const std::vector<GgufTensorInfo> & GgufReader::tensors() const noexcept {
    return tensors_;
}

bool GgufReader::read_tensor(const std::string & name, void * destination,
                             std::size_t bytes) const {
    const GgufTensorInfo * tensor = find_tensor(name);
    if (tensor == nullptr || destination == nullptr ||
        tensor->bytes != bytes || impl_->file == nullptr ||
        tensor->file_offset >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        return false;
    }
    if (fseeko(impl_->file, static_cast<off_t>(tensor->file_offset), SEEK_SET) !=
        0) {
        return false;
    }
    return std::fread(destination, 1, bytes, impl_->file) == bytes;
}

std::vector<float> GgufReader::read_f32_tensor(
    const std::string & name) const {
    const GgufTensorInfo & info = require_tensor(name);
    if (info.dtype != DType::f32) {
        artifact_error("GGUF tensor has the wrong dtype", name,
                       "expected F32");
    }
    if (info.elements > std::numeric_limits<std::size_t>::max() ||
        info.elements > std::numeric_limits<std::uint64_t>::max() /
                            sizeof(float) ||
        info.bytes != info.elements * sizeof(float)) {
        artifact_error("GGUF F32 tensor size is invalid", name,
                       "descriptor overflow or byte mismatch");
    }
    if (info.file_offset >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        artifact_error("GGUF tensor offset is not seekable", name,
                       "offset too large");
    }

    std::vector<float> values(static_cast<std::size_t>(info.elements));
    if (!read_tensor(name, values.data(),
                     static_cast<std::size_t>(info.bytes))) {
        artifact_error("cannot read GGUF tensor payload", name, "I/O failure");
    }
    return values;
}

void GgufReader::require_shape(
    const std::string & name,
    const std::vector<std::int64_t> & expected) const {
    const GgufTensorInfo & tensor = require_tensor(name);
    if (normalized_shape(tensor.shape) != normalized_shape(expected)) {
        artifact_error("GGUF tensor has the wrong shape", name,
                       "expected " + shape_string(expected) + ", got " +
                           shape_string(tensor.shape));
    }
}

} // namespace wam::internal
