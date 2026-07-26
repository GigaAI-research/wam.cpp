#include "models/gwp05/engine/engine_internal.h"

namespace wam::internal::gwp05::engine {

bool run_unrolled_action_denoise(
        Gwp05ModelArch & model, std::vector<float> & action,
        const std::vector<float> & timesteps, const std::vector<float> & sigmas) {
    const Config & cfg = model.cfg;
    const int64_t tokens = cfg.action_chunk;
    if (!prefix_storage_is_valid(model) || !action_prompt_cache_enabled() ||
        !prompt_kv_cache_enabled() ||
        timesteps.size() != static_cast<size_t>(cfg.num_steps) ||
        sigmas.size() != static_cast<size_t>(cfg.num_steps + 1)) {
        return false;
    }
    if (!model.unrolled_action_graph) {
        model.unrolled_action_graph = std::make_unique<UnrolledActionGraph>();
    }
    UnrolledActionGraph & graph = *model.unrolled_action_graph;
    if (!graph.ctx) {
        ggml_init_params params{};
        params.mem_size = 512u * 1024u * 1024u;
        params.no_alloc = true;
        graph.ctx = ggml_init(params);
        if (!graph.ctx) return false;
    }
    ggml_context * ctx = graph.ctx;
    if (!graph.cgraph) {
        const auto graph_build_begin = std::chrono::steady_clock::now();
        graph.action_input = ggml_new_tensor_2d(
            ctx, native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
            cfg.max_action_dim, tokens);
        const size_t bytes = ggml_backend_buft_get_alloc_size(
            ggml_backend_get_default_buffer_type(model.backend), graph.action_input);
        graph.action_input_buffer = ggml_backend_alloc_buffer(model.backend, bytes);
        if (!graph.action_input_buffer ||
            ggml_backend_tensor_alloc(
                graph.action_input_buffer, graph.action_input,
                ggml_backend_buffer_get_base(graph.action_input_buffer)) != GGML_STATUS_SUCCESS) {
            return false;
        }
        graph.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
        ggml_set_input(graph.positions);
        ggml_tensor * step_action = graph.action_input;
        const int64_t condition_tokens = single_token_timestep_enabled(model) ? 1 : tokens;
        for (int step = 0; step < cfg.num_steps; ++step) {
            ggml_tensor * frequency = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, 256, condition_tokens);
            ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            ggml_set_input(frequency);
            ggml_set_input(dt);
            graph.frequency_inputs.push_back(frequency);
            graph.dt_inputs.push_back(dt);
            CachedActionBody body = build_cached_action_body(
                ctx, model, step_action, frequency, graph.positions, dt, nullptr, false);
            step_action = body.action_output;
        }
        graph.action_output = step_action;
        ggml_set_output(graph.action_output);
        graph.cgraph = ggml_new_graph_custom(ctx, 131072, false);
        ggml_build_forward_expand(graph.cgraph, graph.action_output);
        ggml_graph_assign_uid(graph.cgraph);
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend));
        if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, graph.cgraph)) return false;
        std::fprintf(stderr, "wam(gwp05): unrolled action graph nodes=%d buffer=%.1f MiB\n",
                    ggml_graph_n_nodes(graph.cgraph),
                    ggml_gallocr_get_buffer_size(graph.alloc, 0) / (1024.0 * 1024.0));
        model.stats.ms_action_graph_build = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - graph_build_begin).count();
    }

    set_f32_tensor(graph.action_input, action);
    const std::vector<int32_t> positions =
        gwp05_semantics::action_positions(tokens, false);
    ggml_backend_tensor_set(
        graph.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
    for (int step = 0; step < cfg.num_steps; ++step) {
        const std::vector<float> embedding = timestep_embedding(timesteps[step], 256);
        std::vector<float> frequencies;
        if (single_token_timestep_enabled(model)) {
            frequencies = embedding;
        } else {
            frequencies.resize(static_cast<size_t>(256) * tokens);
            for (int64_t i = 0; i < tokens; ++i) {
                std::copy(
                    embedding.begin(), embedding.end(), frequencies.begin() + i * 256);
            }
        }
        const float dt = sigmas[step + 1] - sigmas[step];
        ggml_backend_tensor_set(
            graph.frequency_inputs[step], frequencies.data(), 0,
            frequencies.size() * sizeof(float));
        ggml_backend_tensor_set(graph.dt_inputs[step], &dt, 0, sizeof(dt));
    }
    if (ggml_backend_graph_compute(model.backend, graph.cgraph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    get_f32_tensor(graph.action_output, action);
    return true;
}

bool run_cached_action_step(Gwp05ModelArch & model,
                            const std::vector<float> & action,
                            const std::vector<float> & prompt,
                            float timestep, float dt, bool upload_action,
                            std::vector<float> * prediction_host,
                            std::vector<float> * action_host,
                            int step) {
    NativeActionRegion native_region(model);
    const Config & cfg = model.cfg;
    const int64_t tokens = cfg.action_chunk;
    if (!prefix_storage_is_valid(model)) return false;
    if (!model.cached_action_graph) model.cached_action_graph = std::make_unique<CachedActionGraph>();
    CachedActionGraph & graph = *model.cached_action_graph;
    if (!graph.ctx) {
        ggml_init_params params{}; params.mem_size = 256u * 1024u * 1024u; params.no_alloc = true;
        graph.ctx = ggml_init(params); if (!graph.ctx) return false;
    }
    ggml_context * ctx = graph.ctx;
    const bool build = graph.cgraph == nullptr;
    if (build) {
        const auto graph_build_begin = std::chrono::steady_clock::now();
        graph.action_input = ggml_new_tensor_2d(
            ctx, native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
            cfg.max_action_dim, tokens);
        const size_t bytes = ggml_backend_buft_get_alloc_size(
            ggml_backend_get_default_buffer_type(model.backend), graph.action_input);
        graph.action_input_buffer = ggml_backend_alloc_buffer(model.backend, bytes);
        if (!graph.action_input_buffer || ggml_backend_tensor_alloc(
                graph.action_input_buffer, graph.action_input,
                ggml_backend_buffer_get_base(graph.action_input_buffer)) != GGML_STATUS_SUCCESS) return false;
        const bool cache_action_prompt = action_prompt_cache_enabled();
        const bool cache_prompt_kv = prompt_kv_cache_enabled();
        const bool single_token_timestep = single_token_timestep_enabled(model);
        const int64_t condition_tokens = single_token_timestep ? 1 : tokens;
        if (cache_action_prompt && !model.prefix_storage->action_prompt) return false;
        if (cache_prompt_kv &&
            model.prefix_storage->prompt_keys.size() != static_cast<size_t>(cfg.n_layers)) {
            return false;
        }
        if (!cache_action_prompt) {
            graph.prompt_input = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, cfg.t5_hidden, cfg.n_lang);
        }
        graph.frequency_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 256, condition_tokens);
        graph.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
        graph.dt_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        for (ggml_tensor * input : {graph.action_input, graph.frequency_input,
                                    graph.positions, graph.dt_input}) ggml_set_input(input);
        if (graph.prompt_input) ggml_set_input(graph.prompt_input);
        ggml_tensor * hidden = action_encoder(ctx, model, "action_encoder", graph.action_input);
        ConditionOutput condition = build_condition(
            ctx, model, "action_condition_embedder", graph.frequency_input,
            graph.prompt_input, cfg.expert_h, condition_tokens,
            cache_action_prompt ? model.prefix_storage->action_prompt : nullptr,
            !cache_prompt_kv);
        if (cache_debug_dump_enabled()) {
            graph.action_condition_debug = ggml_dup(ctx, condition.modulation);
            graph.action_temb_debug = ggml_dup(ctx, condition.temb);
        }
        for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
            const std::string prefix = "gwp.blocks." + std::to_string(layer) + ".action_expert";
            ggml_tensor * table = model.weight(prefix + ".scale_shift_table");
            ggml_tensor * combined = modulation_add(ctx, model, condition.modulation, table);
            ggml_tensor * shift = modulation_part(
                ctx, combined, 0, cfg.expert_h, condition_tokens);
            ggml_tensor * scale = modulation_part(
                ctx, combined, 1, cfg.expert_h, condition_tokens);
            ggml_tensor * gate = modulation_part(
                ctx, combined, 2, cfg.expert_h, condition_tokens);
            ggml_tensor * cshift = modulation_part(
                ctx, combined, 3, cfg.expert_h, condition_tokens);
            ggml_tensor * cscale = modulation_part(
                ctx, combined, 4, cfg.expert_h, condition_tokens);
            ggml_tensor * cgate = modulation_part(
                ctx, combined, 5, cfg.expert_h, condition_tokens);
            ExpertQkv qkv = expert_qkv(
                ctx, model, prefix, hidden, scale, shift, graph.positions,
                nullptr, nullptr, nullptr, false);
            if (layer == 0 && cache_debug_dump_enabled()) {
                graph.action_q_debug = ggml_dup(ctx, qkv.q);
                graph.action_k_debug = ggml_dup(ctx, qkv.k);
                graph.action_v_debug = ggml_dup(ctx, qkv.v);
            }
            ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, qkv.q, 0, 2, 1, 3));
            ggml_tensor * action_k = ggml_cont(ctx, ggml_permute(ctx, qkv.k, 0, 2, 1, 3));
            ggml_tensor * action_v = ggml_cont(ctx, ggml_permute(ctx, qkv.v, 1, 2, 0, 3));
            ggml_tensor * state_k = ggml_view_3d(
                ctx, model.prefix_storage->keys[layer], cfg.head_dim, 1, cfg.n_q_heads,
                model.prefix_storage->keys[layer]->nb[1],
                model.prefix_storage->keys[layer]->nb[2], 0);
            ggml_tensor * reference_k = ggml_view_3d(
                ctx, model.prefix_storage->keys[layer], cfg.head_dim, cfg.n_img, cfg.n_q_heads,
                model.prefix_storage->keys[layer]->nb[1],
                model.prefix_storage->keys[layer]->nb[2],
                model.prefix_storage->keys[layer]->nb[1]);
            ggml_tensor * state_v = ggml_view_3d(
                ctx, model.prefix_storage->values[layer], 1, cfg.head_dim, cfg.n_q_heads,
                model.prefix_storage->values[layer]->nb[1],
                model.prefix_storage->values[layer]->nb[2], 0);
            ggml_tensor * reference_v = ggml_view_3d(
                ctx, model.prefix_storage->values[layer], cfg.n_img, cfg.head_dim, cfg.n_q_heads,
                model.prefix_storage->values[layer]->nb[1],
                model.prefix_storage->values[layer]->nb[2],
                model.prefix_storage->values[layer]->nb[0]);
            state_k = cache_compute_tensor(ctx, state_k, native_action_region(model));
            reference_k = cache_compute_tensor(ctx, reference_k, native_action_region(model));
            state_v = cache_compute_tensor(
                ctx, state_v, bf16_attention_value_output(model));
            reference_v = cache_compute_tensor(
                ctx, reference_v, bf16_attention_value_output(model));
            ggml_tensor * joint_k = ggml_concat(
                ctx, ggml_concat(ctx, state_k, action_k, 1), reference_k, 1);
            ggml_tensor * joint_v = ggml_concat(
                ctx, ggml_concat(ctx, state_v, action_v, 0), reference_v, 0);
            const float attention_scale =
                1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
            ggml_tensor * attended = native_attention_context(
                ctx, model, Q, joint_k, joint_v, attention_scale);
            if (!attended) {
                ggml_tensor * scores = ggml_mul_mat(ctx, joint_k, Q);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                ggml_tensor * probs = ggml_soft_max_ext(
                    ctx, scores, nullptr, attention_scale, 0.0f);
                ggml_tensor * attended_heads = ggml_mul_mat(ctx, joint_v, probs);
                attended = ggml_reshape_2d(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, attended_heads, 0, 2, 1, 3)),
                    cfg.hidden, tokens);
            }
            attended = linear_with_activation(
                ctx, model.weight(prefix + ".attn1.to_out.0.weight"),
                model.weight(prefix + ".attn1.to_out.0.bias"), attended,
                bf16_attention_value_output(model));
            hidden = gated_residual(ctx, model, hidden, attended, gate);
            ggml_tensor * prompt_k = cache_prompt_kv
                ? cache_compute_tensor(
                      ctx, model.prefix_storage->prompt_keys[layer],
                      native_action_region(model))
                : nullptr;
            ggml_tensor * prompt_v = cache_prompt_kv
                ? cache_compute_tensor(
                      ctx, model.prefix_storage->prompt_values[layer],
                      bf16_attention_value_output(model))
                : nullptr;
            hidden = expert_cross_ffn(
                ctx, model, prefix, hidden, condition.text,
                cshift, cscale, cgate,
                prompt_k, prompt_v);
            if (layer == 0 && cache_debug_dump_enabled()) {
                graph.block0_action_debug = ggml_dup(ctx, hidden);
            }
            if (layer + 1 == cfg.n_layers && cache_debug_dump_enabled()) {
                graph.block_last_action_debug = ggml_dup(ctx, hidden);
            }
        }
        ggml_tensor * temb = ggml_reshape_3d(
            ctx, condition.temb, cfg.expert_h, 1, condition_tokens);
        ggml_tensor * modulation_shape = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, cfg.expert_h, 2, condition_tokens);
        ggml_tensor * action_scale_shift = model.weight("gwp.action_scale_shift_table");
        if (native_action_region(model)) {
            action_scale_shift = ggml_cast(ctx, action_scale_shift, GGML_TYPE_F32);
        }
        ggml_tensor * table = single_token_timestep
            ? ggml_reshape_3d(
                  ctx, action_scale_shift, cfg.expert_h, 2, 1)
            : ggml_repeat(
                  ctx, action_scale_shift, modulation_shape);
        ggml_tensor * modulation = modulation_add(ctx, model, table, temb);
        ggml_tensor * shift = ggml_view_2d(ctx, modulation, cfg.expert_h, condition_tokens,
                                           modulation->nb[2], 0);
        ggml_tensor * scale = ggml_view_2d(ctx, modulation, cfg.expert_h, condition_tokens,
                                           modulation->nb[2], modulation->nb[1]);
        ggml_tensor * normalized = modulated_norm(
            ctx, hidden, scale, shift, cfg.norm_eps);
        graph.prediction = action_decoder(ctx, model, normalized);
        if (!native_action_region(model)) {
            graph.prediction = ggml_cast(ctx, graph.prediction, GGML_TYPE_F32);
        }
        graph.action_output = scheduler_step(
            ctx, model, graph.action_input, graph.prediction, graph.dt_input);
        ggml_set_output(graph.action_output);
        const bool keep_prediction = debug_dump_enabled() ||
                                     cache_debug_dump_enabled() ||
                                     std::getenv("WAM_GWP05_CPU_SCHEDULER") != nullptr;
        if (keep_prediction) ggml_set_output(graph.prediction);
        if (graph.block0_action_debug) ggml_set_output(graph.block0_action_debug);
        if (graph.block_last_action_debug) ggml_set_output(graph.block_last_action_debug);
        if (graph.action_condition_debug) ggml_set_output(graph.action_condition_debug);
        if (graph.action_temb_debug) ggml_set_output(graph.action_temb_debug);
        if (graph.action_q_debug) ggml_set_output(graph.action_q_debug);
        if (graph.action_k_debug) ggml_set_output(graph.action_k_debug);
        if (graph.action_v_debug) ggml_set_output(graph.action_v_debug);
        graph.cgraph = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(graph.cgraph, graph.action_output);
        if (keep_prediction) ggml_build_forward_expand(graph.cgraph, graph.prediction);
        if (graph.block0_action_debug) {
            ggml_build_forward_expand(graph.cgraph, graph.block0_action_debug);
        }
        if (graph.block_last_action_debug) {
            ggml_build_forward_expand(graph.cgraph, graph.block_last_action_debug);
        }
        if (graph.action_condition_debug) {
            ggml_build_forward_expand(graph.cgraph, graph.action_condition_debug);
        }
        for (ggml_tensor * tensor : {graph.action_temb_debug, graph.action_q_debug,
                                     graph.action_k_debug, graph.action_v_debug}) {
            if (tensor) ggml_build_forward_expand(graph.cgraph, tensor);
        }
        ggml_graph_assign_uid(graph.cgraph);
        graph.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, graph.cgraph)) return false;
        std::fprintf(stderr, "wam(gwp05): cached action graph nodes=%d buffer=%.1f MiB\n",
                    ggml_graph_n_nodes(graph.cgraph),
                    ggml_gallocr_get_buffer_size(graph.alloc, 0) / (1024.0 * 1024.0));
        model.stats.ms_action_graph_build = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - graph_build_begin).count();
    }
    if (upload_action || build) set_f32_tensor(graph.action_input, action);
    const std::vector<float> embedding = timestep_embedding(timestep, 256);
    std::vector<float> frequencies;
    if (single_token_timestep_enabled(model)) {
        frequencies = embedding;
    } else {
        frequencies.resize(static_cast<size_t>(256) * tokens);
        for (int64_t i = 0; i < tokens; ++i) {
            std::copy(embedding.begin(), embedding.end(), frequencies.begin() + i * 256);
        }
    }
    const std::vector<int32_t> positions =
        gwp05_semantics::action_positions(tokens, false);
    if (graph.prompt_input) {
        ggml_backend_tensor_set(
            graph.prompt_input, prompt.data(), 0, prompt.size() * sizeof(float));
    }
    ggml_backend_tensor_set(graph.frequency_input, frequencies.data(), 0, frequencies.size() * sizeof(float));
    ggml_backend_tensor_set(graph.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(graph.dt_input, &dt, 0, sizeof(dt));
    if (ggml_backend_graph_compute(model.backend, graph.cgraph) != GGML_STATUS_SUCCESS) return false;
    if (graph.block0_action_debug) {
        char name[64];
        std::snprintf(name, sizeof(name), "denoise_%02d_block0_action", step);
        debug_dump_tensor(name, graph.block0_action_debug);
    }
    if (graph.block_last_action_debug) {
        char name[64];
        std::snprintf(name, sizeof(name), "denoise_%02d_block_last_action", step);
        debug_dump_tensor(name, graph.block_last_action_debug);
    }
    if (graph.action_condition_debug) {
        char name[64];
        std::snprintf(name, sizeof(name), "denoise_%02d_action_condition", step);
        debug_dump_tensor(name, graph.action_condition_debug);
    }
    if (graph.action_temb_debug) {
        char name[64];
        std::snprintf(name, sizeof(name), "denoise_%02d_action_temb", step);
        debug_dump_tensor(name, graph.action_temb_debug);
    }
    if (step == 0) {
        debug_dump_tensor("denoise_00_action_q", graph.action_q_debug);
        debug_dump_tensor("denoise_00_action_k", graph.action_k_debug);
        debug_dump_tensor("denoise_00_action_v", graph.action_v_debug);
    }
    if (prediction_host) {
        get_f32_tensor(graph.prediction, *prediction_host);
    }
    ggml_backend_tensor_copy(graph.action_output, graph.action_input);
    ggml_backend_synchronize(model.backend);
    if (action_host) {
        get_f32_tensor(graph.action_input, *action_host);
    }
    return true;
}
} // namespace wam::internal::gwp05::engine
