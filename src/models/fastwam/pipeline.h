#pragma once

#include "backends/ggml/debug_dump.h"
#include "models/fastwam/contract.h"
#include "models/fastwam/inputs.h"
#include "runtime/logger.h"
#include "runtime/runtime_types.h"

#include <memory>

namespace wam::internal::fastwam {

class ModelResources;
struct SessionState;

struct ModelOptions {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    int device_index = 0;
    std::shared_ptr<runtime::Logger> logger;
    std::shared_ptr<ggml_backend::DebugDump> debug_dump;
};

struct LoadedModel {
    std::shared_ptr<ModelResources> resources;
    runtime::EngineInfo info;
};

LoadedModel load_model_resources(const FastWamContract & contract,
                                 const ModelOptions & options);
std::unique_ptr<SessionState> create_session_state();
CoreAction run_pipeline(ModelResources & resources, SessionState & state,
                        const FastWamContract & contract,
                        const PreparedInputs & inputs);
void reset_session(ModelResources & resources, SessionState & state);

} // namespace wam::internal::fastwam
