#pragma once

#include "models/gwp05/contract.h"
#include "models/gwp05/inputs.h"
#include "models/gwp05/types.h"

#include "backends/ggml/debug_dump.h"
#include "runtime/logger.h"
#include "runtime/runtime_types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace wam::internal::gwp05 {

struct ModelResources;
struct SessionState;

struct ModelOptions {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    int device_index = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::tokens;
    std::optional<FixedPrompt> fixed_prompt;
    RuntimeTuningConfig tuning;
    std::shared_ptr<runtime::Logger> logger;
    std::shared_ptr<ggml_backend::DebugDump> debug_dump;
};

struct LoadedModel {
    std::shared_ptr<ModelResources> resources;
    runtime::EngineInfo info;
};

KernelDispatch resolve_kernel_dispatch(
    ComputePrecision precision, Backend backend,
    const std::string & conversion_policy);
const char * execution_profile_name(ExecutionProfile profile) noexcept;

LoadedModel load_model_resources(const Gwp05Contract & contract,
                                 const ModelOptions & options);
std::unique_ptr<SessionState> create_session_state(
    const ModelResources & resources);
CoreAction run_pipeline(ModelResources & resources, SessionState & state,
                        const PreparedInputs & inputs,
                        bool enable_prefix_cache);
void reset_session(ModelResources & resources, SessionState & state);

} // namespace wam::internal::gwp05
