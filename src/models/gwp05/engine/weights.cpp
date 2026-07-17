
constexpr float kPi = 3.14159265358979323846f;

enum class MotPrecisionPolicy {
    F32,
    NATIVE_BF16,
};

const char * precision_policy_name(MotPrecisionPolicy policy) {
    return policy == MotPrecisionPolicy::NATIVE_BF16 ? "native-bf16" : "f32";
}

const char * debug_dump_directory() {
    const char * directory = std::getenv("WAM_GWP05_CACHE_DUMP_DIR");
    if (directory && *directory) return directory;
    return std::getenv("WAM_GWP05_DUMP_DIR");
}

void debug_dump(const char * name, const std::vector<float> & values,
                const std::vector<int64_t> & shape) {
    const char * directory = debug_dump_directory();
    if (!directory || !*directory) return;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    const std::filesystem::path path = std::filesystem::path(directory) / (std::string(name) + ".f32");
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::fprintf(stderr, "wam(gwp05): cannot write debug dump %s\n", path.c_str());
        return;
    }
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        std::fprintf(stderr, "wam(gwp05): cannot write debug dump %s\n", path.c_str());
        return;
    }
    const std::filesystem::path metadata_path =
        std::filesystem::path(directory) / (std::string(name) + ".json");
    std::ofstream metadata(metadata_path);
    if (!metadata) {
        std::fprintf(stderr, "wam(gwp05): cannot write debug metadata %s\n",
                     metadata_path.c_str());
        return;
    }
    const uint16_t byte_order_probe = 1;
    const bool little_endian = *reinterpret_cast<const uint8_t *>(&byte_order_probe) == 1;
    metadata << "{\"format\":\"gwp-f32-dump-v1\",\"dtype\":\"float32\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) metadata << ',';
        metadata << shape[i];
    }
    metadata << "],\"elements\":" << values.size()
             << ",\"byte_order\":\"" << (little_endian ? "little" : "big") << "\"}\n";
}

bool debug_dump_enabled() {
    const char * directory = std::getenv("WAM_GWP05_DUMP_DIR");
    return directory && *directory;
}

bool cache_debug_dump_enabled() {
    const char * directory = std::getenv("WAM_GWP05_CACHE_DUMP_DIR");
    return directory && *directory;
}

bool any_debug_dump_enabled() {
    return debug_dump_enabled() || cache_debug_dump_enabled();
}

void audit_mixed_binary_nodes(const char * graph_name, ggml_cgraph * graph) {
    if (!std::getenv("WAM_GWP05_AUDIT_DTYPES") || !graph) return;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (!node->src[0] || !node->src[1] ||
            node->src[0]->type == node->src[1]->type) continue;
        if (node->op != GGML_OP_ADD && node->op != GGML_OP_SUB &&
            node->op != GGML_OP_MUL && node->op != GGML_OP_DIV) continue;
        std::fprintf(
            stderr,
            "wam(gwp05): dtype audit graph=%s node=%d op=%s dst=%s src0=%s(%s) src1=%s(%s)\n",
            graph_name, i, ggml_op_name(node->op), ggml_type_name(node->type),
            ggml_type_name(node->src[0]->type), ggml_op_name(node->src[0]->op),
            ggml_type_name(node->src[1]->type), ggml_op_name(node->src[1]->op));
    }
}

void debug_dump_tensor(const char * name, ggml_tensor * tensor) {
    if (!any_debug_dump_enabled() || !tensor ||
        !tensor->buffer ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_BF16)) return;
    std::vector<float> values(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    } else {
        std::vector<ggml_bf16_t> raw(values.size());
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(raw.data(), values.data(), values.size());
    }
    std::vector<int64_t> shape;
    for (int dimension = ggml_n_dims(tensor) - 1; dimension >= 0; --dimension) {
        shape.push_back(tensor->ne[dimension]);
    }
    debug_dump(name, values, shape);
}

void replace_all(std::string & value, const char * source, const char * target) {
    const size_t source_size = std::strlen(source);
    const size_t target_size = std::strlen(target);
    size_t position = 0;
    while ((position = value.find(source, position)) != std::string::npos) {
        value.replace(position, source_size, target);
        position += target_size;
    }
}

std::string compact_tensor_name(const std::string & full_name) {
    const size_t dot = full_name.find('.');
    if (dot == std::string::npos) return full_name;
    const std::string component = full_name.substr(0, dot);
    std::string name = full_name.substr(dot + 1);
    auto apply = [&](std::initializer_list<std::pair<const char *, const char *>> replacements) {
        for (const auto & item : replacements) replace_all(name, item.first, item.second);
    };
    if (component == "gwp") {
        apply({
            {"action_condition_embedder.", "acond."}, {"condition_embedder.", "cond."},
            {"patch_embedding.", "patch."}, {"state_encoder.", "state."},
            {"action_encoder.", "action."}, {"action_decoder.", "decode."},
            {"blocks.", "blk."}, {".action_expert.", ".a."},
            {".visual_expert.", ".v."}, {"time_embedder.linear_1.", "time.0."},
            {"time_embedder.linear_2.", "time.1."}, {"text_embedder.linear_1.", "text.0."},
            {"text_embedder.linear_2.", "text.1."}, {"time_proj.", "time.out."},
            {".ffn.net.0.proj.", ".ffn.in."}, {".ffn.net.2.", ".ffn.out."},
            {".attn1.", ".sa."}, {".attn2.", ".ca."},
            {".to_out.0.", ".o."}, {".to_qkv.", ".qkv."}, {".to_q.", ".q."},
            {".to_k.", ".k."}, {".to_v.", ".vproj."},
            {".norm_q.", ".qn."}, {".norm_k.", ".kn."},
            {".norm2.", ".ca_norm."}, {"in_proj.", "in."},
            {"mid_proj.", "mid."}, {"out_proj.", "out."},
            {"scale_shift_table", "mod"},
        });
    } else if (component == "t5") {
        apply({
            {"shared.", "token_embd."}, {"encoder.final_layer_norm.", "final_norm."},
            {"encoder.block.", "blk."}, {".layer.0.SelfAttention.", ".sa."},
            {".layer.0.layer_norm.", ".sa_norm."},
            {".layer.1.DenseReluDense.", ".ffn."},
            {".layer.1.layer_norm.", ".ffn_norm."},
            {"relative_attention_bias.", "rel."}, {"wi_0.", "wi0."},
            {"wi_1.", "wi1."},
        });
    } else if (component == "vae") {
        apply({
            {"encoder.", "enc."}, {"down_blocks.", "down."},
            {"downsampler.resample.1.", "ds."}, {"resnets.", "res."},
            {"mid_block.", "mid."}, {"attentions.0.", "attn."},
            {"conv_shortcut.", "skip."}, {"conv_in.", "in."},
            {"conv_out.", "out."}, {"norm_out.gamma", "out_norm"},
            {"to_qkv.", "qkv."}, {"quant_conv.", "quant."},
            {"norm1.gamma", "n1"}, {"norm2.gamma", "n2"},
            {"norm.gamma", "norm"}, {"conv1.", "c1."}, {"conv2.", "c2."},
        });
    } else {
        return full_name;
    }
    return component + "." + name;
}

struct GraphContext {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;

    ~GraphContext() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

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

bool action_prompt_cache_enabled() {
    return std::getenv("WAM_GWP05_DISABLE_ACTION_PROMPT_CACHE") == nullptr;
}

bool prompt_kv_cache_enabled() {
    return action_prompt_cache_enabled() &&
           std::getenv("WAM_GWP05_DISABLE_PROMPT_KV_CACHE") == nullptr;
}

size_t prompt_cache_capacity() {
    const char * value = std::getenv("WAM_GWP05_PROMPT_CACHE_SIZE");
    if (!value || !*value) return 4;
    if (std::strchr(value, '-')) return 0;
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || !end || *end ||
        parsed > std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return static_cast<size_t>(parsed);
}

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

struct Gwp05ModelArch final : Engine {
    Gwp05ModelArch() = default;
    ~Gwp05ModelArch() override {
        mot_graph.reset();
        unrolled_action_graph.reset();
        cached_action_graph.reset();
        prompt_projection_graph.reset();
        prefix_graph.reset();
        prefix_storage.reset();
        vae_graph.reset();
#ifdef WAM_GWP05_CUDA_PREPROCESS
        gwp05_cuda_preprocessor_destroy(cuda_preprocessor);
#endif
        for (ggml_backend_buffer_t & buffer : weight_buffers) {
            if (buffer) ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        for (ggml_context *& context : weight_contexts) {
            if (context) ggml_free(context);
            context = nullptr;
        }
        if (backend) ggml_backend_free(backend);
    }

    std::vector<float> predict(const EngineInputsView & in) override;
    void reset() override;

    ggml_backend_t backend = nullptr;
    int device_index = 0;
    // MoT, UMT5, and VAE own independent contexts and backend allocations.
    // Graphs access tensors through the non-owning name index below.
    std::array<ggml_backend_buffer_t, 3> weight_buffers{};
    std::array<ggml_context *, 3> weight_contexts{};
    std::array<bool, 3> component_loaded{};
    std::array<std::uint64_t, 3> component_device_bytes{};
    std::array<double, 3> component_load_milliseconds{};
    std::array<double, 3> component_unload_milliseconds{};
    LoadState load_state = LoadState::empty;
    bool metadata_only = false;
    MotPrecisionPolicy precision_policy = MotPrecisionPolicy::F32;
    KernelDispatch dispatch{};
    std::string conversion_policy;
    bool building_native_action = false;
    int n_threads = default_cpu_threads();
    std::unordered_map<std::string, ggml_tensor *> weights;
    std::unique_ptr<MotGraph> mot_graph;
    std::unique_ptr<VaeGraph> vae_graph;
#ifdef WAM_GWP05_CUDA_PREPROCESS
    Gwp05CudaPreprocessor * cuda_preprocessor = nullptr;
#endif
    std::unique_ptr<PrefixGraph> prefix_graph;
    std::unique_ptr<PrefixStorage> prefix_storage;
    std::unique_ptr<PromptProjectionGraph> prompt_projection_graph;
    std::unique_ptr<CachedActionGraph> cached_action_graph;
    std::unique_ptr<UnrolledActionGraph> unrolled_action_graph;
    std::list<PromptCacheEntry> prompt_cache;
    std::optional<PromptCacheEntry> fixed_prompt;
    size_t prompt_cache_limit = prompt_cache_capacity();
    uint64_t prompt_cache_hits = 0;
    uint64_t prompt_cache_misses = 0;
    std::vector<float> projected_prompt_signature;
    uint64_t projected_prompt_hits = 0;
    uint64_t projected_prompt_misses = 0;
    int64_t t5_max_length = 0;

    std::vector<float> state_q01;
    std::vector<float> state_q99;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
    std::vector<float> vae_latents_mean;
    std::vector<float> vae_latents_std;
    std::mt19937 rng{std::random_device{}()};

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
            std::fprintf(stderr, "wam(gwp05): internal missing weight %s\n", name.c_str());
            return nullptr;
        }
        return it->second;
    }
};

bool single_token_timestep_enabled(const Gwp05ModelArch & model) {
    return model.dispatch.single_token_timestep;
}

bool cross_request_prompt_cache_enabled(const Gwp05ModelArch & model) {
    return prompt_kv_cache_enabled() && model.prompt_cache_limit != 0;
}

bool native_bf16(const Gwp05ModelArch & model) {
    return model.dispatch.native_bf16;
}

bool native_vae_bf16(const Gwp05ModelArch & model) {
    return model.dispatch.bf16_vae;
}

bool packed_self_qkv(const Gwp05ModelArch & model) {
    return model.dispatch.packed_qkv;
}

bool native_action_region(const Gwp05ModelArch & model) {
    return native_bf16(model) && model.building_native_action;
}

struct NativeActionRegion {
    explicit NativeActionRegion(Gwp05ModelArch & model_) : model(model_) {
        previous = model.building_native_action;
        model.building_native_action = native_bf16(model);
    }
    ~NativeActionRegion() { model.building_native_action = previous; }
    Gwp05ModelArch & model;
    bool previous = false;
};

bool tensor_shape_is(const ggml_tensor * tensor, const char * name,
                     std::initializer_list<int64_t> shape) {
    if (!tensor) {
        std::fprintf(stderr, "wam(gwp05): missing runtime tensor %s\n", name);
        return false;
    }
    size_t dimension = 0;
    for (int64_t expected : shape) {
        if (tensor->ne[dimension] != expected) {
            std::fprintf(stderr,
                         "wam(gwp05): runtime tensor %s dimension %zu is %lld, expected %lld\n",
                         name, dimension, static_cast<long long>(tensor->ne[dimension]),
                         static_cast<long long>(expected));
            return false;
        }
        ++dimension;
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] != 1) {
            std::fprintf(stderr, "wam(gwp05): runtime tensor %s has an unexpected dimension %zu\n",
                         name, dimension);
            return false;
        }
    }
    return true;
}

bool bf16_attention_value_output(const Gwp05ModelArch & model);
bool bf16_kv_cache_storage(const Gwp05ModelArch & model);

bool prefix_storage_is_valid(const Gwp05ModelArch & model) {
    const Config & cfg = model.cfg;
    const PrefixStorage * storage = model.prefix_storage.get();
    if (!storage) {
        std::fprintf(stderr, "wam(gwp05): cached action requested without prefix storage\n");
        return false;
    }
    const size_t layers = static_cast<size_t>(cfg.n_layers);
    const ggml_type expected_key_type = bf16_kv_cache_storage(model)
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    const ggml_type expected_value_type = (bf16_kv_cache_storage(model) ||
                                           bf16_attention_value_output(model))
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    if (storage->keys.size() != layers || storage->values.size() != layers) {
        std::fprintf(stderr, "wam(gwp05): prefix K/V layer count does not match model layers\n");
        return false;
    }
    if (action_prompt_cache_enabled() &&
        !tensor_shape_is(storage->action_prompt, "projected action prompt",
                         {cfg.expert_h, cfg.n_lang})) {
        return false;
    }
    if (action_prompt_cache_enabled() && storage->action_prompt->type !=
            (native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32)) {
        std::fprintf(stderr, "wam(gwp05): projected action prompt dtype mismatches policy\n");
        return false;
    }
    if (prompt_kv_cache_enabled() &&
        (storage->prompt_keys.size() != layers || storage->prompt_values.size() != layers)) {
        std::fprintf(stderr, "wam(gwp05): prompt K/V layer count does not match model layers\n");
        return false;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        if (!tensor_shape_is(storage->keys[layer], "prefix key",
                             {cfg.head_dim, cfg.n_prefix, cfg.n_q_heads}) ||
            !tensor_shape_is(storage->values[layer], "prefix value",
                             {cfg.n_prefix, cfg.head_dim, cfg.n_q_heads}) ||
            storage->keys[layer]->type != expected_key_type ||
            storage->values[layer]->type != expected_value_type) {
            return false;
        }
        if (prompt_kv_cache_enabled() &&
            (!tensor_shape_is(storage->prompt_keys[layer], "prompt key",
                              {cfg.head_dim, cfg.n_lang, cfg.n_q_heads}) ||
             !tensor_shape_is(storage->prompt_values[layer], "prompt value",
                              {cfg.n_lang, cfg.head_dim, cfg.n_q_heads}) ||
             storage->prompt_keys[layer]->type != expected_key_type ||
             storage->prompt_values[layer]->type != expected_value_type)) {
            return false;
        }
    }
    return true;
}

bool init_backend(Gwp05ModelArch & model) {
#ifdef GGML_USE_CUDA
    model.backend = ggml_backend_cuda_init(model.device_index);
    if (model.backend) std::fprintf(stderr, "wam(gwp05): using CUDA backend\n");
#endif
#ifdef GGML_USE_METAL
    if (!model.backend) model.backend = ggml_backend_metal_init();
    if (model.backend) std::fprintf(stderr, "wam(gwp05): using Metal backend\n");
#endif
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
        if (!model.backend) return false;
        ggml_backend_cpu_set_n_threads(model.backend, model.n_threads);
        std::fprintf(stderr, "wam(gwp05): using CPU backend (%d threads; debug only)\n", model.n_threads);
    }
    const char * backend_name = ggml_backend_name(model.backend);
    const bool cuda = backend_name && std::strstr(backend_name, "CUDA") != nullptr;
    if (native_bf16(model) && !cuda) {
        std::fprintf(stderr,
                     "wam(gwp05): native-bf16 requires the validated CUDA backend; "
                     "no implicit mixed or CPU fallback is allowed\n");
        return false;
    }
#ifndef WAM_GWP05_CUDNN
    if (native_vae_bf16(model)) {
        std::fprintf(stderr,
                     "wam(gwp05): BF16 VAE policy requires a build configured "
                     "with WAM_GWP05_CUDNN=ON\n");
        return false;
    }
#endif
    std::fprintf(stderr,
        "wam(gwp05): execution precision=%s weights=%s hidden=%s cache=%s "
        "vae=%s math=%s attention=%s denoise=%s "
        "graph_identity=stable fallback=none\n",
        precision_policy_name(model.precision_policy),
        native_bf16(model) ? "BF16" : "F32",
        native_bf16(model) ? "BF16" : "F32",
        native_bf16(model) ? "BF16" : "F32",
        native_vae_bf16(model) ? "BF16-convolution/F32-norm" : "F32",
        "f32-sensitive-islands", "explicit",
        model.dispatch.unrolled_denoise ? "single-token-unrolled" : "one-step-replay");
    return true;
}

const char * component_prefix(WeightComponent component) {
    switch (component) {
        case WeightComponent::mot: return "gwp.";
        case WeightComponent::t5: return "t5.";
        case WeightComponent::vae: return "vae.";
    }
    return "";
}

const char * component_name(WeightComponent component) {
    switch (component) {
        case WeightComponent::mot: return "MoT";
        case WeightComponent::t5: return "UMT5";
        case WeightComponent::vae: return "VAE";
    }
    return "unknown";
}

void refresh_component_telemetry(Gwp05ModelArch & model) {
    model.runtime_components.clear();
    model.resident_device_bytes = 0;
    for (WeightComponent component : {WeightComponent::mot, WeightComponent::t5,
                                      WeightComponent::vae}) {
        const size_t index = static_cast<size_t>(component);
        model.runtime_components.push_back({component_name(component),
                                            model.component_loaded[index],
                                            model.component_device_bytes[index],
                                            model.component_load_milliseconds[index],
                                            model.component_unload_milliseconds[index]});
        if (model.component_loaded[index]) {
            model.resident_device_bytes += model.component_device_bytes[index];
        }
    }
    model.text_encoder_resident =
        model.component_loaded[static_cast<size_t>(WeightComponent::t5)];
    model.peak_component_device_bytes = std::max(
        model.peak_component_device_bytes, model.resident_device_bytes);
}

bool component_is_loaded(const Gwp05ModelArch & model, WeightComponent component) {
    return model.component_loaded[static_cast<size_t>(component)];
}

void unload_component(Gwp05ModelArch & model, WeightComponent component) {
    const auto begin = std::chrono::steady_clock::now();
    const size_t index = static_cast<size_t>(component);
    if (!model.component_loaded[index] && !model.weight_buffers[index] &&
        !model.weight_contexts[index]) {
        return;
    }
    if (model.backend) ggml_backend_synchronize(model.backend);
    const std::string prefix = component_prefix(component);
    for (auto it = model.weights.begin(); it != model.weights.end();) {
        if (it->first.compare(0, prefix.size(), prefix) == 0) {
            it = model.weights.erase(it);
        } else {
            ++it;
        }
    }
    if (model.weight_buffers[index]) {
        ggml_backend_buffer_free(model.weight_buffers[index]);
        model.weight_buffers[index] = nullptr;
    }
    if (model.weight_contexts[index]) {
        ggml_free(model.weight_contexts[index]);
        model.weight_contexts[index] = nullptr;
    }
    model.component_loaded[index] = false;
    model.component_device_bytes[index] = 0;
    model.component_unload_milliseconds[index] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    refresh_component_telemetry(model);
}

bool load_component(GgufReader & reader, Gwp05ModelArch & model,
                    WeightComponent component) {
    const auto begin = std::chrono::steady_clock::now();
    const size_t index = static_cast<size_t>(component);
    if (model.component_loaded[index]) return true;
    const auto & tensors = reader.tensors();
    const char * prefix = component_prefix(component);
    const size_t prefix_size = std::strlen(prefix);
    size_t weight_count = 0;
    for (const GgufTensorInfo & info : tensors) {
        if (std::strncmp(info.name.c_str(), prefix, prefix_size) == 0) ++weight_count;
    }
    if (weight_count == 0) return false;
    ggml_init_params params{};
    params.mem_size = (weight_count + 32) * ggml_tensor_overhead();
    params.no_alloc = true;
    model.weight_contexts[index] = ggml_init(params);
    if (!model.weight_contexts[index]) return false;

    size_t max_tensor_bytes = 0;
    for (const GgufTensorInfo & info : tensors) {
        const char * name = info.name.c_str();
        if (std::strncmp(name, prefix, prefix_size) != 0) continue;
        const ggml_tensor * source = reader.descriptor(name);
        ggml_tensor * destination = ggml_new_tensor(
            model.weight_contexts[index], source->type,
            ggml_n_dims(source), source->ne);
        ggml_set_name(destination, name);
        model.weights.emplace(name, destination);
        max_tensor_bytes = std::max(max_tensor_bytes, ggml_nbytes(destination));
    }
    model.weight_buffers[index] =
        ggml_backend_alloc_ctx_tensors(model.weight_contexts[index], model.backend);
    if (!model.weight_buffers[index]) {
        std::fprintf(stderr, "wam(gwp05): cannot allocate %s weight buffer\n",
                     component_name(component));
        unload_component(model, component);
        return false;
    }

    std::vector<uint8_t> staging(max_tensor_bytes);
    size_t loaded = 0;
    for (auto & item : model.weights) {
        if (item.first.compare(0, prefix_size, prefix) != 0) continue;
        ggml_tensor * destination = item.second;
        const size_t bytes = ggml_nbytes(destination);
        if (!reader.read_tensor(item.first, staging.data(), bytes)) {
            unload_component(model, component);
            return false;
        }
        ggml_backend_tensor_set(destination, staging.data(), 0, bytes);
        if ((++loaded % 100) == 0) {
            std::fprintf(stderr, "wam(gwp05): loaded %s %zu/%zu tensors\r",
                         component_name(component), loaded, weight_count);
            std::fflush(stderr);
        }
    }
    const size_t bytes = ggml_backend_buffer_get_size(model.weight_buffers[index]);
    model.component_loaded[index] = true;
    model.component_device_bytes[index] = bytes;
    model.component_load_milliseconds[index] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    refresh_component_telemetry(model);
    std::fprintf(stderr, "wam(gwp05): %s owns %zu tensors (%.2f GiB)\n",
                 component_name(component), weight_count,
                 static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return true;
}

bool load_resident_weights(GgufReader & reader, Gwp05ModelArch & model) {
    if (!load_component(reader, model, WeightComponent::mot) ||
        !load_component(reader, model, WeightComponent::t5) ||
        !load_component(reader, model, WeightComponent::vae)) {
        model.load_state = LoadState::failed;
        return false;
    }
    model.load_state = LoadState::fully_resident;
    std::fprintf(stderr, "wam(gwp05): loaded %zu tensors (%.2f GiB total)\n",
                 model.weights.size(), static_cast<double>(model.resident_device_bytes) /
                     (1024.0 * 1024.0 * 1024.0));
    return true;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                     ggml_tensor * input) {
    ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
    if (bias && bias->type != result->type) {
        bias = ggml_cast(ctx, bias, result->type);
    }
    return bias ? ggml_add(ctx, result, bias) : result;
}

bool bf16_projection_activations(const Gwp05ModelArch & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_ffn_activations(const Gwp05ModelArch & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_qkv_activations(const Gwp05ModelArch & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_attention_value_output(const Gwp05ModelArch & model) {
    return native_action_region(model);
}

bool bf16_kv_cache_storage(const Gwp05ModelArch & model) {
    return model.dispatch.bf16_hidden_and_kv;
}

bool unsafe_fast_native_bf16(const Gwp05ModelArch & model) {
    return native_action_region(model) && !model.dispatch.f32_accumulation;
}

ggml_tensor * linear_with_activation(ggml_context * ctx, ggml_tensor * weight,
                                     ggml_tensor * bias, ggml_tensor * input,
                                     bool use_bf16) {
    if (use_bf16 && input->type != GGML_TYPE_BF16) {
        input = ggml_cast(ctx, input, GGML_TYPE_BF16);
    }
    if (use_bf16 && weight->type == GGML_TYPE_BF16) {
        ggml_tensor * result = ggml_mul_mat_out_type(
            ctx, weight, input, GGML_TYPE_BF16);
        return bias ? ggml_add(ctx, result, bias) : result;
    }
    return linear(ctx, weight, bias, input);
}

ggml_tensor * linear_gelu_with_activation(ggml_context * ctx,
                                          ggml_tensor * weight,
                                          ggml_tensor * bias,
                                          ggml_tensor * input,
                                          bool use_bf16) {
    if (use_bf16 && weight->type == GGML_TYPE_BF16 && bias &&
        bias->type == GGML_TYPE_BF16) {
        if (input->type != GGML_TYPE_BF16) {
            input = ggml_cast(ctx, input, GGML_TYPE_BF16);
        }
        ggml_tensor * projected = ggml_mul_mat_out_type(
            ctx, weight, input, GGML_TYPE_BF16);
        return ggml_bias_gelu_bf16(ctx, projected, bias);
    }
    return ggml_gelu(ctx, linear_with_activation(
        ctx, weight, bias, input, use_bf16));
}

ggml_tensor * cache_compute_tensor(ggml_context * ctx, ggml_tensor * tensor,
                                   bool use_bf16) {
    if (!tensor) return nullptr;
    const ggml_type target = use_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    return tensor->type == target ? tensor : ggml_cast(ctx, tensor, target);
}

void set_f32_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(
            tensor, values.data(), 0, values.size() * sizeof(float));
        return;
    }
    GGML_ASSERT(tensor->type == GGML_TYPE_BF16);
    std::vector<ggml_bf16_t> converted(values.size());
    ggml_fp32_to_bf16_row(values.data(), converted.data(), values.size());
    ggml_backend_tensor_set(
        tensor, converted.data(), 0, converted.size() * sizeof(ggml_bf16_t));
}

void get_f32_tensor(ggml_tensor * tensor, std::vector<float> & values) {
    values.resize(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(
            tensor, values.data(), 0, values.size() * sizeof(float));
        return;
    }
    GGML_ASSERT(tensor->type == GGML_TYPE_BF16);
    std::vector<ggml_bf16_t> raw(values.size());
    ggml_backend_tensor_get(
        tensor, raw.data(), 0, raw.size() * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row(raw.data(), values.data(), values.size());
}

ggml_tensor * scheduler_step(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * action, ggml_tensor * velocity,
                             ggml_tensor * dt) {
    if (!native_action_region(model)) {
        return ggml_add(ctx, action, ggml_mul(ctx, velocity, dt));
    }
    if (unsafe_fast_native_bf16(model)) {
        if (action->type != GGML_TYPE_BF16) action = ggml_cast(ctx, action, GGML_TYPE_BF16);
        if (velocity->type != GGML_TYPE_BF16) velocity = ggml_cast(ctx, velocity, GGML_TYPE_BF16);
        if (dt->type != GGML_TYPE_BF16) dt = ggml_cast(ctx, dt, GGML_TYPE_BF16);
        return ggml_add(ctx, action, ggml_mul(ctx, velocity, dt));
    }
    ggml_tensor * action_f32 = ggml_cast(ctx, action, GGML_TYPE_F32);
    ggml_tensor * velocity_f32 = ggml_cast(ctx, velocity, GGML_TYPE_F32);
    return ggml_cast(
        ctx, ggml_add(ctx, action_f32, ggml_mul(ctx, velocity_f32, dt)),
        GGML_TYPE_BF16);
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps) {
    if (input->type == GGML_TYPE_BF16 && weight && weight->type == GGML_TYPE_BF16) {
        return ggml_rms_norm_bf16_f32(ctx, input, weight, eps);
    }
    if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * result = ggml_rms_norm(ctx, input, eps);
    if (!weight) return result;
    ggml_tensor * scale = ggml_cast(ctx, weight, GGML_TYPE_F32);
    if (ggml_n_dims(input) == 3 && ggml_n_dims(scale) == 1) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, 1);
    }
    return ggml_mul(ctx, result, scale);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps) {
    if (input->type == GGML_TYPE_BF16 && weight && bias &&
        weight->type == GGML_TYPE_BF16 && bias->type == GGML_TYPE_BF16) {
        return ggml_layer_norm_bf16(ctx, input, weight, bias, eps);
    }
    if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * result = ggml_norm(ctx, input, eps);
    if (weight) result = ggml_mul(ctx, result, ggml_cast(ctx, weight, GGML_TYPE_F32));
    if (bias) result = ggml_add(ctx, result, ggml_cast(ctx, bias, GGML_TYPE_F32));
    return result;
}
