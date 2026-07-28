#pragma once

#include "models/gwp05/engine/engine.h"
#include "artifact/gguf_reader.h"
#include "backends/ggml/backend_context.h"
#include "backends/ggml/debug_dump.h"
#include "backends/ggml/graph_context.h"
#include "backends/ggml/weight_store.h"
#include "models/common/scheduler.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wam::internal::gwp05::engine {

namespace gwp05_semantics = ::wam::internal::gwp05::semantics;
using GraphContext = ggml_backend::GraphContext;

constexpr float kPi = 3.14159265358979323846F;

enum class LanguageExecutionMode {
    tokens,
    fixed_tokens,
    external_embedding,
};

enum class MotPrecisionPolicy {
    F32,
    NATIVE_BF16,
};

struct Config {
    std::int64_t n_img = 0;
    std::int64_t n_lang = 0;
    std::int64_t n_state = 0;
    std::int64_t n_prefix = 0;
    std::int64_t n_suffix = 0;
    std::int64_t n_full = 0;
    std::int64_t hidden = 0;
    std::int64_t expert_h = 0;
    std::int64_t intermediate = 0;
    std::int64_t expert_inter = 0;
    std::int64_t n_q_heads = 0;
    std::int64_t n_kv_heads = 0;
    std::int64_t head_dim = 0;
    std::int64_t q_full_dim = 0;
    std::int64_t kv_full_dim = 0;
    std::int64_t n_layers = 0;
    std::int64_t max_state_dim = 0;
    std::int64_t max_action_dim = 0;
    std::int64_t real_state_dim = 0;
    std::int64_t real_action_dim = 0;
    float norm_eps = 0.0F;
    int num_steps = 0;
    int action_chunk = 0;
    float rms_eps = 0.0F;
    int rope_n_dims = 0;
    float rope_freq_base = 0.0F;
    std::int64_t image_height = 0;
    std::int64_t image_width = 0;
    std::int64_t num_views = 0;
    std::int64_t t5_hidden = 0;
    std::int64_t t5_ffn_dim = 0;
    std::int64_t t5_heads = 0;
    std::int64_t t5_head_dim = 0;
    std::int64_t t5_layers = 0;
    std::int64_t t5_vocab_size = 0;
    std::int64_t vae_z_dim = 0;
    std::int64_t num_embodiments = 0;
    std::int64_t embodiment_id = 0;
    float flow_shift = 0.0F;
};

struct EngineInputsView {
    const float * composite_image = nullptr;
    int composite_image_n = 0;
    const float * precomputed_ref_latent = nullptr;
    int precomputed_ref_latent_n = 0;
    const float * precomputed_prompt_emb = nullptr;
    int precomputed_prompt_tokens = 0;
    const std::int32_t * lang_tokens = nullptr;
    int n_lang = 0;
    const float * model_state = nullptr;
    const float * action_noise = nullptr;
    const std::int32_t * attention_mask = nullptr;
    int attention_mask_n = 0;
    bool enable_prefix_cache = true;
};

struct EngineTelemetry {
    float ms_total = 0.0F;
    float ms_vision = 0.0F;
    float ms_inference = 0.0F;
    float ms_prefill = 0.0F;
    float ms_denoise = 0.0F;
    float ms_vae = 0.0F;
    float ms_vae_preprocess = 0.0F;
    float ms_vae_graph = 0.0F;
    float ms_umt5 = 0.0F;
    float ms_prefix_cache = 0.0F;
    float ms_prefix_graph_build = 0.0F;
    float ms_action_graph_build = 0.0F;
    float ms_prompt_projection = 0.0F;
    std::vector<float> ms_denoise_steps;
    bool prompt_cache_hit = false;
    bool prompt_cache_enabled = false;
    std::uint64_t prompt_cache_hits = 0;
    std::uint64_t prompt_cache_misses = 0;
    std::size_t prompt_cache_entries = 0;
    std::size_t prompt_cache_bytes = 0;
    bool projected_prompt_cache_hit = false;
    std::uint64_t projected_prompt_hits = 0;
    std::uint64_t projected_prompt_misses = 0;
};

int default_cpu_threads();
std::string compact_tensor_name(const std::string & full_name);

struct MotGraph : GraphContext {
    ~MotGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }

    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * state_input = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * reference_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_frequency = nullptr;
    ggml_tensor * visual_frequency = nullptr;
    ggml_tensor * action_positions = nullptr;
    ggml_tensor * visual_time = nullptr;
    ggml_tensor * visual_height = nullptr;
    ggml_tensor * visual_width = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * dt_input = nullptr;
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
    ggml_tensor * action_tokens_debug = nullptr;
    ggml_tensor * visual_tokens_debug = nullptr;
    ggml_tensor * action_condition_debug = nullptr;
    ggml_tensor * block0_action_debug = nullptr;
};

struct VaeGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * pixels = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * conv_in_debug = nullptr;
    ggml_tensor * down0_debug = nullptr;
    ggml_tensor * down1_debug = nullptr;
    ggml_tensor * down2_debug = nullptr;
    ggml_tensor * down3_debug = nullptr;
};

struct PrefixGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * state_input = nullptr;
    ggml_tensor * reference_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_frequency = nullptr;
    ggml_tensor * visual_frequency = nullptr;
    ggml_tensor * state_position = nullptr;
    ggml_tensor * visual_time = nullptr;
    ggml_tensor * visual_height = nullptr;
    ggml_tensor * visual_width = nullptr;
    ggml_tensor * action_prompt = nullptr;
    ggml_tensor * state_hidden_in = nullptr;
    ggml_tensor * visual_hidden_in = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
    std::vector<ggml_tensor *> state_hidden_out;
    std::vector<ggml_tensor *> visual_hidden_out;
};

struct PrefixStorage {
    ~PrefixStorage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * action_prompt = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
};

struct PromptProjectionGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_prompt = nullptr;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
};

struct CachedActionGraph : GraphContext {
    ~CachedActionGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * frequency_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * dt_input = nullptr;
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_tensor * block0_action_debug = nullptr;
    ggml_tensor * block_last_action_debug = nullptr;
    ggml_tensor * action_condition_debug = nullptr;
    ggml_tensor * action_temb_debug = nullptr;
    ggml_tensor * action_q_debug = nullptr;
    ggml_tensor * action_k_debug = nullptr;
    ggml_tensor * action_v_debug = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
};

struct UnrolledActionGraph : GraphContext {
    ~UnrolledActionGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * positions = nullptr;
    std::vector<ggml_tensor *> frequency_inputs;
    std::vector<ggml_tensor *> dt_inputs;
    ggml_tensor * action_output = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
};


struct PromptCacheEntry {
    std::vector<int32_t> tokens;
    std::vector<int32_t> mask;
    std::vector<float> embedding;
};

enum class WeightComponent : std::size_t {
    mot = 0,
    t5 = 1,
    vae = 2,
};

enum class LoadState {
    empty,
    text_only,
    compute_only,
    fully_resident,
    failed,
};

struct Gwp05ModelArch {
    Gwp05ModelArch() = default;
    ~Gwp05ModelArch();

    std::vector<float> predict_core(const EngineInputsView & in);
    void reset_core();

    Config cfg{};
    EngineTelemetry stats{};
    std::vector<RuntimeComponentInfo> runtime_components;
    LanguageExecutionMode language_mode = LanguageExecutionMode::tokens;
    bool text_encoder_resident = false;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;

    ggml_backend_t backend = nullptr;
    ggml_backend::BackendContext backend_context;
    Backend backend_request = Backend::automatic;
    int device_index = 0;
    // MoT, UMT5, and VAE own independent contexts and backend allocations.
    // Graphs access tensors through the non-owning name index below.
    std::array<std::unique_ptr<ggml_backend::WeightStore>, 3> weight_stores{};
    std::array<bool, 3> component_loaded{};
    std::array<std::uint64_t, 3> component_device_bytes{};
    std::array<double, 3> component_load_milliseconds{};
    std::array<double, 3> component_unload_milliseconds{};
    LoadState load_state = LoadState::empty;
    bool metadata_only = false;
    MotPrecisionPolicy precision_policy = MotPrecisionPolicy::F32;
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
    size_t prompt_cache_limit = 0;
    uint64_t prompt_cache_hits = 0;
    uint64_t prompt_cache_misses = 0;
    std::vector<float> projected_prompt_signature;
    uint64_t projected_prompt_hits = 0;
    uint64_t projected_prompt_misses = 0;
    int64_t t5_max_length = 0;

    std::vector<float> vae_latents_mean;
    std::vector<float> vae_latents_std;
    std::vector<int32_t> prompt_mask(const EngineInputsView & in) const {
        if (in.attention_mask && in.attention_mask_n == in.n_lang) {
            return {in.attention_mask, in.attention_mask + in.n_lang};
        }
        return std::vector<int32_t>(static_cast<size_t>(in.n_lang), 1);
    }

    bool get_cached_prompt(const EngineInputsView & in, std::vector<float> & output) {
        if (!prompt_cache_limit || !in.lang_tokens || in.n_lang < 1) return false;
        const std::vector<int32_t> mask = prompt_mask(in);
        for (auto it = prompt_cache.begin(); it != prompt_cache.end(); ++it) {
            if (it->tokens.size() != static_cast<size_t>(in.n_lang) || it->mask != mask ||
                !std::equal(it->tokens.begin(), it->tokens.end(), in.lang_tokens)) {
                continue;
            }
            output = it->embedding;
            prompt_cache.splice(prompt_cache.begin(), prompt_cache, it);
            ++prompt_cache_hits;
            return true;
        }
        ++prompt_cache_misses;
        return false;
    }

    void put_cached_prompt(const EngineInputsView & in, const std::vector<float> & embedding) {
        if (!prompt_cache_limit) return;
        PromptCacheEntry entry;
        entry.tokens.assign(in.lang_tokens, in.lang_tokens + in.n_lang);
        entry.mask = prompt_mask(in);
        entry.embedding = embedding;
        prompt_cache.push_front(std::move(entry));
        while (prompt_cache.size() > prompt_cache_limit) prompt_cache.pop_back();
    }

    size_t prompt_cache_embedding_bytes() const {
        size_t bytes = 0;
        for (const PromptCacheEntry & entry : prompt_cache) {
            bytes += entry.embedding.size() * sizeof(float);
        }
        return bytes;
    }

    ggml_tensor * weight(const std::string & name) const {
        const auto it = weights.find(compact_tensor_name(name));
        if (it == weights.end()) {
            if (logger) {
                logger->logf(LogLevel::error,
                             "gwp05: internal missing weight %s", name.c_str());
            }
            return nullptr;
        }
        return it->second;
    }
};

struct EngineResources final : Gwp05ModelArch {
    ~EngineResources();
};

struct EngineSessionState final : Gwp05ModelArch {};

class Engine final {
public:
    explicit Engine(std::unique_ptr<EngineResources> resources,
                    EngineInfo info)
        : resources_(std::move(resources)), info_(std::move(info)) {}

private:
    friend EngineSessionPtr create_engine_session(
        Engine &, const EngineSessionOptions &);
    friend CoreAction predict(EngineSession &, const PreparedInputs &);
    friend void reset(EngineSession &);
    friend const EngineInfo & engine_info(const Engine &) noexcept;
    friend class EngineSession;

    std::unique_ptr<EngineResources> resources_;
    EngineInfo info_;
    std::mutex execution_mutex_;
};

class EngineSession final {
public:
    EngineSession(Engine & owner, std::unique_ptr<EngineSessionState> state,
                  bool enable_prefix_cache)
        : owner_(&owner), state_(std::move(state)),
          enable_prefix_cache_(enable_prefix_cache) {}

private:
    friend CoreAction predict(EngineSession &, const PreparedInputs &);
    friend void reset(EngineSession &);

    Engine * owner_ = nullptr;
    std::unique_ptr<EngineSessionState> state_;
    bool enable_prefix_cache_ = true;
};

bool native_bf16(const Gwp05ModelArch & model);

struct NativeActionRegion {
    explicit NativeActionRegion(Gwp05ModelArch & model_) : model(model_) {
        previous = model.building_native_action;
        model.building_native_action = native_bf16(model);
    }
    ~NativeActionRegion() { model.building_native_action = previous; }
    Gwp05ModelArch & model;
    bool previous = false;
};

struct ConditionOutput {
    ggml_tensor * temb = nullptr;
    ggml_tensor * modulation = nullptr;
    ggml_tensor * text = nullptr;
};
struct CrossKv {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct ExpertQkv {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct CachedActionBody {
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_tensor * block0_action = nullptr;
    ggml_tensor * action_condition = nullptr;
};


const char * precision_policy_name(MotPrecisionPolicy policy);
void logf(const Gwp05ModelArch & model, LogLevel level,
          const char * format, ...);
void debug_dump(const Gwp05ModelArch & model, const char * name,
                const std::vector<float> & values,
                const std::vector<std::int64_t> & shape);
bool debug_dump_enabled(const Gwp05ModelArch & model);
bool cache_debug_dump_enabled(const Gwp05ModelArch & model);
bool any_debug_dump_enabled(const Gwp05ModelArch & model);
void audit_mixed_binary_nodes(const Gwp05ModelArch & model,
                              const char * graph_name, ggml_cgraph * graph);
void debug_dump_tensor(const Gwp05ModelArch & model, const char * name,
                       ggml_tensor * tensor);

bool action_prompt_cache_enabled(const Gwp05ModelArch & model);
bool prompt_kv_cache_enabled(const Gwp05ModelArch & model);
bool single_token_timestep_enabled(const Gwp05ModelArch & model);
bool cross_request_prompt_cache_enabled(const Gwp05ModelArch & model);
bool native_bf16(const Gwp05ModelArch & model);
bool native_vae_bf16(const Gwp05ModelArch & model);
bool packed_self_qkv(const Gwp05ModelArch & model);
bool native_action_region(const Gwp05ModelArch & model);
bool tensor_shape_is(const ggml_tensor * tensor, const char * name,
                     std::initializer_list<std::int64_t> shape);
bool prefix_storage_is_valid(const Gwp05ModelArch & model);
bool init_backend(Gwp05ModelArch & model);
void refresh_component_telemetry(Gwp05ModelArch & model);
bool component_is_loaded(const Gwp05ModelArch & model, WeightComponent component);
void unload_component(Gwp05ModelArch & model, WeightComponent component);
bool load_component(GgufReader & reader, Gwp05ModelArch & model,
                    WeightComponent component);
bool load_resident_weights(GgufReader & reader, Gwp05ModelArch & model);

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight,
                     ggml_tensor * bias, ggml_tensor * input);
bool bf16_projection_activations(const Gwp05ModelArch & model);
bool bf16_ffn_activations(const Gwp05ModelArch & model);
bool bf16_qkv_activations(const Gwp05ModelArch & model);
bool bf16_attention_value_output(const Gwp05ModelArch & model);
bool bf16_kv_cache_storage(const Gwp05ModelArch & model);
bool unsafe_fast_native_bf16(const Gwp05ModelArch & model);
ggml_tensor * linear_with_activation(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
    ggml_tensor * input, bool use_bf16);
ggml_tensor * linear_gelu_with_activation(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
    ggml_tensor * input, bool use_bf16);
ggml_tensor * build_attention(
    ggml_context * ctx, ggml_tensor * query_input,
    ggml_tensor * key_value_input, ggml_tensor * query_weight,
    ggml_tensor * query_bias, ggml_tensor * key_weight,
    ggml_tensor * key_bias, ggml_tensor * value_weight,
    ggml_tensor * value_bias, ggml_tensor * output_weight,
    ggml_tensor * output_bias, ggml_tensor * query_norm,
    ggml_tensor * key_norm, std::int64_t heads, std::int64_t head_dim,
    ggml_tensor * mask, float scale, float eps,
    bool use_bf16_qkv = false, bool use_bf16_value_output = false);
ggml_tensor * cache_compute_tensor(ggml_context * ctx, ggml_tensor * tensor,
                                   bool use_bf16);
void set_f32_tensor(ggml_tensor * tensor, const std::vector<float> & values);
void get_f32_tensor(ggml_tensor * tensor, std::vector<float> & values);
ggml_tensor * scheduler_step(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * action, ggml_tensor * prediction,
                             ggml_tensor * dt);
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps);

std::vector<float> timestep_embedding(float timestep, std::int64_t dim);
bool validate_prompt_mask(const EngineInputsView & input, int & valid_tokens);
std::vector<float> run_t5(Gwp05ModelArch & model,
                          const EngineInputsView & input);
std::vector<float> run_vae(Gwp05ModelArch & model,
                           const EngineInputsView & input);

ggml_tensor * action_encoder(ggml_context * ctx, Gwp05ModelArch & model,
                             const std::string & prefix, ggml_tensor * input);
ggml_tensor * action_decoder(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * input);
ConditionOutput build_condition(
    ggml_context * ctx, Gwp05ModelArch & model, const std::string & prefix,
    ggml_tensor * frequency, ggml_tensor * prompt, std::int64_t hidden,
    std::int64_t tokens, ggml_tensor * projected_prompt = nullptr,
    bool build_text = true);
CrossKv build_cross_kv(ggml_context * ctx, Gwp05ModelArch & model,
                       const std::string & prefix, ggml_tensor * prompt);
ggml_tensor * modulation_part(ggml_context * ctx, ggml_tensor * combined,
                              int part, std::int64_t hidden,
                              std::int64_t tokens);
ggml_tensor * modulation_add(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * modulation, ggml_tensor * table);
ggml_tensor * modulated_norm(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * scale, ggml_tensor * shift,
                             float epsilon);
ggml_tensor * gated_residual(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * hidden, ggml_tensor * branch,
                             ggml_tensor * gate);
ExpertQkv expert_qkv(
    ggml_context * ctx, Gwp05ModelArch & model, const std::string & prefix,
    ggml_tensor * hidden, ggml_tensor * zero, ggml_tensor * scale,
    ggml_tensor * action_positions, ggml_tensor * visual_time,
    ggml_tensor * visual_height, ggml_tensor * visual_width, bool visual);
ggml_tensor * native_attention_context(
    ggml_context * ctx, Gwp05ModelArch & model, ggml_tensor * query,
    ggml_tensor * key, ggml_tensor * value, float scale);
ggml_tensor * expert_cross_ffn(
    ggml_context * ctx, Gwp05ModelArch & model, const std::string & prefix,
    ggml_tensor * hidden, ggml_tensor * prompt, ggml_tensor * scale,
    ggml_tensor * zero, ggml_tensor * gate, ggml_tensor * cached_key = nullptr,
    ggml_tensor * cached_value = nullptr);

CachedActionBody build_cached_action_body(
    ggml_context * ctx, Gwp05ModelArch & model, ggml_tensor * action_input,
    ggml_tensor * frequency_input, ggml_tensor * positions,
    ggml_tensor * dt_input, ggml_tensor * prompt_input, bool keep_debug);

bool run_unrolled_action_denoise(
    Gwp05ModelArch & model, std::vector<float> & action,
    const std::vector<float> & timesteps, const std::vector<float> & sigmas);
bool run_cached_action_step(
    Gwp05ModelArch & model, const std::vector<float> & action,
    const std::vector<float> & prompt, float timestep, float dt,
    bool upload_action, std::vector<float> * prediction,
    std::vector<float> * action_output, int step_index);
bool run_mot_step(
    Gwp05ModelArch & model, const std::vector<float> & state,
    const std::vector<float> & action, const std::vector<float> & reference,
    const std::vector<float> & prompt, float timestep, float dt,
    bool upload_action, bool upload_prefix, std::vector<float> * prediction,
    std::vector<float> * action_output);
bool build_prefix_cache(
    Gwp05ModelArch & model, const std::vector<float> & state,
    const std::vector<float> & reference, const std::vector<float> & prompt);

} // namespace wam::internal::gwp05::engine
