#pragma once

#include "wam/observation.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wam {

struct Tensor {
    std::vector<std::uint8_t> data;
    DType dtype = DType::unknown;
    std::vector<std::int64_t> shape;
    std::string layout;
    ByteOrder byte_order = ByteOrder::not_applicable;

    bool empty() const noexcept;
};

struct NamedTensor {
    std::string name;
    Tensor tensor;
};

struct PolicyActionChunk : Tensor {
    PolicyActionChunk() = default;
    PolicyActionChunk(Tensor tensor);
    PolicyActionChunk & operator=(Tensor tensor);

    std::size_t horizon() const noexcept;
    std::size_t action_dimension() const noexcept;
};

struct PhaseTiming {
    std::string name;
    double milliseconds = 0.0;
};

struct Telemetry {
    double preprocess_milliseconds = 0.0;
    double model_milliseconds = 0.0;
    double model_vision_milliseconds = 0.0;
    double model_text_milliseconds = 0.0;
    double model_prefill_milliseconds = 0.0;
    double model_decode_milliseconds = 0.0;
    double postprocess_milliseconds = 0.0;
    double total_milliseconds = 0.0;

    std::vector<PhaseTiming> model_timings;
    std::uint64_t peak_device_memory_bytes = 0;
};

struct Prediction {
    PolicyActionChunk action;
    std::vector<NamedTensor> auxiliary;
    Telemetry telemetry;
};

} // namespace wam
