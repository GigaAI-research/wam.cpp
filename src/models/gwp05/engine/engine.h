#pragma once

#include "models/gwp05/artifact.h"
#include "models/gwp05/inputs.h"
#include "models/common/model_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace wam::internal::gwp05 {

namespace engine {

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

enum class ExecutionProfile {
    reference,
    latency,
};

struct KernelDispatch {
    ExecutionProfile profile = ExecutionProfile::reference;
    bool native_bf16 = false;
    bool bf16_output_gemm = false;
    bool f32_accumulation = true;
    bool bf16_hidden_and_kv = false;
    bool bf16_vae = false;
    bool packed_qkv = false;
    bool cuda_graphs = false;
    bool single_token_timestep = false;
    bool unrolled_denoise = false;
};

struct EngineOptions {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    int device_index = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::tokens;
    std::optional<FixedPrompt> fixed_prompt;
};

struct EngineSessionOptions {
    bool enable_prefix_cache = true;
};

using EngineInfo = model_common::EngineInfo;

KernelDispatch resolve_kernel_dispatch(
    ComputePrecision precision, Backend backend,
    const std::string & conversion_policy);
const char * execution_profile_name(ExecutionProfile profile) noexcept;

EnginePtr create_engine(const ArtifactContract & artifact,
                        const EngineOptions & options);
EngineSessionPtr create_engine_session(
    Engine & engine, const EngineSessionOptions & options);
CoreAction predict(EngineSession & session, const PreparedInputs & inputs);
void reset(EngineSession & session);
const EngineInfo & engine_info(const Engine & engine) noexcept;

} // namespace engine
} // namespace wam::internal::gwp05
