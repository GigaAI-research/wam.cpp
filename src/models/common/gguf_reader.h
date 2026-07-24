#pragma once

#include "wam/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam::internal {

struct GgufTensorInfo {
    std::string name;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::uint64_t bytes = 0;
    std::uint64_t file_offset = 0;
};

class GgufReader final {
public:
    static std::shared_ptr<GgufReader> open(const std::string & path);
    ~GgufReader();

    const std::string & path() const noexcept;
    std::uint64_t file_size() const noexcept;
    bool has(const std::string & key) const noexcept;
    std::string require_string(const std::string & key) const;
    std::uint32_t require_u32(const std::string & key) const;
    float require_f32(const std::string & key) const;
    std::vector<std::string> require_string_array(const std::string & key) const;
    const GgufTensorInfo & require_tensor(const std::string & name) const;
    std::vector<float> read_f32_tensor(const std::string & name) const;

private:
    explicit GgufReader(std::string path);

    std::string path_;
    std::uint64_t file_size_ = 0;
    std::vector<GgufTensorInfo> tensors_;
};

} // namespace wam::internal
