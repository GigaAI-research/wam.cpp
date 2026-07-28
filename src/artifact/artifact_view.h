#pragma once

#include "artifact/gguf_reader.h"
#include "artifact/manifest.h"

#include <memory>
#include <string>

namespace wam::internal::artifact {

class ArtifactView final {
public:
    static ArtifactView open(const std::string & input_path);

    const ArtifactBundle & bundle() const noexcept;
    const GgufReader & gguf() const noexcept;
    std::shared_ptr<GgufReader> shared_gguf() const noexcept;

    const std::string & path() const noexcept;
    std::uint64_t file_size() const noexcept;
    bool has(const std::string & key) const noexcept;
    std::string require_string(const std::string & key) const;
    std::uint32_t require_u32(const std::string & key) const;
    std::uint64_t require_u64(const std::string & key) const;
    float require_f32(const std::string & key) const;
    bool require_bool(const std::string & key) const;
    std::vector<float> require_f32_array(const std::string & key) const;
    std::vector<float> optional_f32_array(const std::string & key) const;
    std::vector<std::string> require_string_array(
        const std::string & key) const;
    std::vector<std::uint32_t> require_u32_array(
        const std::string & key) const;
    std::vector<std::int32_t> require_i32_array(
        const std::string & key) const;
    const GgufTensorInfo * find_tensor(const std::string & name) const noexcept;
    const GgufTensorInfo & require_tensor(const std::string & name) const;
    const std::vector<GgufTensorInfo> & tensors() const noexcept;
    std::vector<float> read_f32_tensor(const std::string & name) const;
    void require_shape(const std::string & name,
                       const std::vector<std::int64_t> & shape) const;

private:
    ArtifactView(ArtifactBundle bundle, std::shared_ptr<GgufReader> reader);

    ArtifactBundle bundle_;
    std::shared_ptr<GgufReader> reader_;
};

} // namespace wam::internal::artifact
