#pragma once

#include "models/gwp05/graphs.h"
#include "models/gwp05/types.h"

#include "backends/ggml/backend_context.h"
#include "backends/ggml/debug_dump.h"
#include "backends/ggml/weight_store.h"
#include "runtime/logger.h"
#include "wam/model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace wam::internal::gwp05 {

struct PromptCacheEntry {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> mask;
    std::vector<float> embedding;
};

enum class WeightComponent : std::size_t {
    mot = 0,
    umt5 = 1,
    vision_vae = 2,
};

enum class LoadState {
    empty,
    text_only,
    compute_only,
    fully_resident,
    failed,
};

int default_cpu_threads();
std::string compact_tensor_name(const std::string & full_name);

struct ExecutionState {
    ExecutionState() = default;
    ~ExecutionState();

    void reset();
    std::vector<float> execute(const PipelineInputsView & input);
    std::vector<std::int32_t> prompt_mask(
        const PipelineInputsView & input) const;
    bool get_cached_prompt(const PipelineInputsView & input,
                           std::vector<float> & output);
    void put_cached_prompt(const PipelineInputsView & input,
                           const std::vector<float> & embedding);
    std::size_t prompt_cache_embedding_bytes() const;
    ggml_tensor * weight(const std::string & name) const;

    ModelGeometry cfg{};
    PipelineTelemetry stats{};
    std::vector<RuntimeComponentInfo> runtime_components;
    LanguageExecutionMode language_mode = LanguageExecutionMode::tokens;
    bool text_encoder_resident = false;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;

    ggml_backend_t backend = nullptr;
    Backend backend_request = Backend::automatic;
    int device_index = 0;
    std::array<bool, 3> component_loaded{};
    std::array<std::uint64_t, 3> component_device_bytes{};
    std::array<double, 3> component_load_milliseconds{};
    std::array<double, 3> component_unload_milliseconds{};
    LoadState load_state = LoadState::empty;
    bool metadata_only = false;
    MotPrecisionPolicy precision_policy = MotPrecisionPolicy::f32;
    KernelDispatch dispatch{};
    RuntimeTuningConfig tuning{};
    std::shared_ptr<runtime::Logger> logger;
    std::shared_ptr<ggml_backend::DebugDump> debug_dumper;
    std::string conversion_policy;
    bool building_native_action = false;
    int n_threads = default_cpu_threads();
    std::unordered_map<std::string, ggml_tensor *> weights;

    std::unique_ptr<MotGraph> mot_graph;
    std::unique_ptr<VaeGraph> vae_graph;
    std::unique_ptr<PrefixGraph> prefix_graph;
    std::unique_ptr<PrefixStorage> prefix_storage;
    std::unique_ptr<PromptProjectionGraph> prompt_projection_graph;
    std::unique_ptr<CachedActionGraph> cached_action_graph;
    std::unique_ptr<UnrolledActionGraph> unrolled_action_graph;

    std::list<PromptCacheEntry> prompt_cache;
    std::optional<PromptCacheEntry> fixed_prompt;
    std::size_t prompt_cache_limit = 0;
    std::uint64_t prompt_cache_hits = 0;
    std::uint64_t prompt_cache_misses = 0;
    std::vector<float> projected_prompt_signature;
    std::uint64_t projected_prompt_hits = 0;
    std::uint64_t projected_prompt_misses = 0;
    std::int64_t t5_max_length = 0;

    std::vector<float> vae_latents_mean;
    std::vector<float> vae_latents_std;
};

struct ModelResources final : ExecutionState {
    ~ModelResources();
    ggml_backend::BackendContext backend_context;
    std::array<std::unique_ptr<ggml_backend::WeightStore>, 3> weight_stores{};
    std::mutex execution_mutex;
};

struct SessionState final : ExecutionState {};

} // namespace wam::internal::gwp05
