#pragma once

#include "wam/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam::internal {

struct GgufTensorInfo {
    std::string name;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::uint64_t elements = 0;
    std::uint64_t bytes = 0;
    std::uint64_t file_offset = 0;
};

class GgufReader final {
public:
    static std::shared_ptr<GgufReader> open(const std::string & path);
    ~GgufReader();
    GgufReader(const GgufReader &) = delete;
    GgufReader & operator=(const GgufReader &) = delete;

    const std::string & path() const noexcept;
    std::uint64_t file_size() const noexcept;
    std::uint64_t data_offset() const noexcept;

    bool has(const std::string & key) const noexcept;
    std::string require_string(const std::string & key) const;
    std::string optional_string(const std::string & key,
                                std::string fallback = {}) const;
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

    std::size_t tensor_count() const noexcept;
    const GgufTensorInfo * find_tensor(const std::string & name) const noexcept;
    const GgufTensorInfo & require_tensor(const std::string & name) const;
    const std::vector<GgufTensorInfo> & tensors() const noexcept;
    bool read_tensor(const std::string & name, void * destination,
                     std::size_t bytes) const;
    std::vector<float> read_f32_tensor(const std::string & name) const;
    void require_shape(const std::string & name,
                       const std::vector<std::int64_t> & shape) const;

private:
    struct Impl;

    explicit GgufReader(std::string path);
    void initialize();

    std::string path_;
    std::uint64_t file_size_ = 0;
    std::uint64_t data_offset_ = 0;
    std::vector<GgufTensorInfo> tensors_;
    std::unique_ptr<Impl> impl_;
};

} // namespace wam::internal
