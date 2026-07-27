#pragma once

#include "models/fastwam/artifact.h"
#include "models/fastwam/inputs.h"
#include "models/common/model_types.h"

#include <memory>

namespace wam::internal::fastwam::engine {

class Engine;
class EngineSession;

struct EngineDeleter {
    void operator()(Engine * engine) const noexcept;
};
using EnginePtr = std::unique_ptr<Engine, EngineDeleter>;

struct EngineSessionDeleter {
    void operator()(EngineSession * session) const noexcept;
};
using EngineSessionPtr =
    std::unique_ptr<EngineSession, EngineSessionDeleter>;

struct EngineOptions {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    int device_index = 0;
};

struct EngineSessionOptions {};

using EngineInfo = model_common::EngineInfo;

EnginePtr create_engine(const ArtifactContract & artifact,
                        const EngineOptions & options);
EngineSessionPtr create_engine_session(
    Engine & engine, const EngineSessionOptions & options = {});
CoreAction predict(EngineSession & session, const PreparedInputs & inputs);
void reset(EngineSession & session);
const EngineInfo & engine_info(const Engine & engine) noexcept;

} // namespace wam::internal::fastwam::engine
