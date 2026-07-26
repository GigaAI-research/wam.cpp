#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wam::test {

class MetadataFixture final {
public:
    MetadataFixture();
    ~MetadataFixture();
    MetadataFixture(MetadataFixture &&) noexcept;
    MetadataFixture & operator=(MetadataFixture &&) noexcept;
    MetadataFixture(const MetadataFixture &) = delete;
    MetadataFixture & operator=(const MetadataFixture &) = delete;

    void set_string(std::string key, std::string value);
    void set_u32(std::string key, std::uint32_t value);
    void set_f32(std::string key, float value);
    void set_bool(std::string key, bool value);
    void set_string_array(std::string key, std::vector<std::string> value);
    void set_u32_array(std::string key, std::vector<std::uint32_t> value);
    void set_i32_array(std::string key, std::vector<std::int32_t> value);
    void set_f32_tensor(std::string name, std::vector<float> value);
    void remove(const std::string & key);
    void write(const std::string & path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

MetadataFixture valid_policy_fixture();
MetadataFixture valid_gwp05_policy_fixture();
MetadataFixture valid_gwp05_robotwin_14d_policy_fixture();
MetadataFixture valid_gwp05_legacy_fixture();

} // namespace wam::test
