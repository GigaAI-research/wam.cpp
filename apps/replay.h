#pragma once

#include "wam/wam.h"

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "json_fwd.hpp"

namespace wam::apps {

struct OwnedImage {
    std::string name;
    ImageEncoding encoding = ImageEncoding::unknown;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::size_t row_stride_bytes = 0;
    std::vector<std::uint8_t> data;
};

struct OwnedTensor {
    DType dtype = DType::unknown;
    ByteOrder byte_order = ByteOrder::little;
    std::vector<std::int64_t> shape;
    std::string layout;
    std::vector<std::uint8_t> data;

    TensorView view() const;
};

struct ReplayRequest {
    std::string format;
    std::vector<OwnedImage> images;
    OwnedTensor prompt_embedding;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
    OwnedTensor state;
    OwnedTensor noise;

    Inputs inputs() const;
};

struct Comparison {
    bool present = false;
    bool finite = false;
    std::size_t elements = 0;
    double mean_abs = 0.0;
    double max_abs = 0.0;
    std::string expected_sha256;
};

ReplayRequest load_request(const std::filesystem::path & path);
Precision parse_precision(const std::string & value);
std::string precision_name(Precision precision);
LanguageEncoderPolicy parse_language_policy(const std::string & value);
std::string language_policy_name(LanguageEncoderPolicy policy);
std::string sha256_hex(const void * data, std::size_t size);
std::vector<float> action_values(const Tensor & action);
Comparison compare_action(const Tensor & action, const std::filesystem::path & expected);
void write_action(const Tensor & action, const std::filesystem::path & path,
                  const std::string & format);

nlohmann::json stats_json(const Stats & stats);
nlohmann::json model_json(const ModelInfo & info);
nlohmann::json device_json(std::int32_t device_index);
nlohmann::json comparison_json(const Comparison & comparison);
nlohmann::json action_json(const Tensor & action, bool include_values);
void write_json(const nlohmann::json & value, const std::string & path);

double percentile(std::vector<double> values, double quantile);
nlohmann::json sample_summary(const std::vector<double> & values);
long peak_rss_kib();

} // namespace wam::apps
