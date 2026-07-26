#pragma once

#include "models/fastwam/artifact.h"
#include "models/fastwam/inputs.h"

#include <memory>

namespace wam::internal::fastwam::engine {

class Engine;

struct EngineDeleter {
    void operator()(Engine * engine) const noexcept;
};
using EnginePtr = std::unique_ptr<Engine, EngineDeleter>;

struct EngineOptions {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    int device_index = 0;
};

struct EngineInfo {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::unknown;
    std::uint64_t resident_device_bytes = 0;
    std::vector<RuntimeComponentInfo> runtime_components;
};

EnginePtr create_engine(const ArtifactContract & artifact,
                        const EngineOptions & options);
CoreAction predict(Engine & engine, const ArtifactContract & artifact,
                   const PreparedInputs & inputs);
void reset(Engine & engine);
const EngineInfo & engine_info(const Engine & engine) noexcept;

} // namespace wam::internal::fastwam::engine
