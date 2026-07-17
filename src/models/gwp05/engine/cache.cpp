bool ensure_prefix_storage(Gwp05ModelArch & model) {
    if (model.prefix_storage) return true;
    const Config & cfg = model.cfg;
    model.prefix_storage = std::make_unique<PrefixStorage>();
    PrefixStorage & storage = *model.prefix_storage;
    ggml_init_params params{};
    params.mem_size = 8u * 1024u * 1024u;
    params.no_alloc = true;
    storage.ctx = ggml_init(params);
    if (!storage.ctx) return false;
    const ggml_type key_type = bf16_kv_cache_storage(model)
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    const ggml_type value_type = (bf16_kv_cache_storage(model) ||
                                  bf16_attention_value_output(model))
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    if (action_prompt_cache_enabled()) {
        storage.action_prompt = ggml_new_tensor_2d(
            storage.ctx, native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
            cfg.expert_h, cfg.n_lang);
    }
    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        storage.keys.push_back(ggml_new_tensor_3d(
            storage.ctx, key_type, cfg.head_dim, cfg.n_prefix, cfg.n_q_heads));
        storage.values.push_back(ggml_new_tensor_3d(
            storage.ctx, value_type, cfg.n_prefix, cfg.head_dim, cfg.n_q_heads));
        if (prompt_kv_cache_enabled()) {
            storage.prompt_keys.push_back(ggml_new_tensor_3d(
                storage.ctx, key_type, cfg.head_dim, cfg.n_lang, cfg.n_q_heads));
            storage.prompt_values.push_back(ggml_new_tensor_3d(
                storage.ctx, value_type, cfg.n_lang, cfg.head_dim, cfg.n_q_heads));
        }
    }
    storage.buffer = ggml_backend_alloc_ctx_tensors(storage.ctx, model.backend);
    if (!storage.buffer) return false;
    std::fprintf(stderr, "wam(gwp05): prefix cache %.1f MiB\n",
                ggml_backend_buffer_get_size(storage.buffer) / (1024.0 * 1024.0));
    if (storage.action_prompt) {
        std::fprintf(stderr, "wam(gwp05): action prompt cache %.1f MiB\n",
                    ggml_nbytes(storage.action_prompt) / (1024.0 * 1024.0));
    }
    if (!storage.prompt_keys.empty()) {
        double prompt_kv_mib = 0.0;
        for (size_t index = 0; index < storage.prompt_keys.size(); ++index) {
            prompt_kv_mib += static_cast<double>(ggml_nbytes(storage.prompt_keys[index]));
            prompt_kv_mib += static_cast<double>(ggml_nbytes(storage.prompt_values[index]));
        }
        std::fprintf(stderr, "wam(gwp05): prompt K/V cache %.1f MiB\n",
                    prompt_kv_mib / (1024.0 * 1024.0));
    }
    return true;
}

bool update_projected_prompt_cache(Gwp05ModelArch & model,
                                   const std::vector<float> & prompt) {
    if (!cross_request_prompt_cache_enabled(model)) return true;
    const auto begin = std::chrono::steady_clock::now();
    const bool hit = model.projected_prompt_signature.size() == prompt.size() &&
        std::equal(prompt.begin(), prompt.end(), model.projected_prompt_signature.begin());
    model.stats.projected_prompt_cache_hit = hit;
    if (hit) {
        ++model.projected_prompt_hits;
        model.stats.projected_prompt_hits = model.projected_prompt_hits;
        model.stats.projected_prompt_misses = model.projected_prompt_misses;
        model.stats.ms_prompt_projection = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        return true;
    }
    ++model.projected_prompt_misses;
    if (!ensure_prefix_storage(model)) return false;
    if (!model.prompt_projection_graph) {
        model.prompt_projection_graph = std::make_unique<PromptProjectionGraph>();
    }
    PromptProjectionGraph & graph = *model.prompt_projection_graph;
    const Config & cfg = model.cfg;
    if (!graph.ctx) {
        ggml_init_params params{};
        params.mem_size = 64u * 1024u * 1024u;
        params.no_alloc = true;
        graph.ctx = ggml_init(params);
        if (!graph.ctx) return false;
    }
    if (!graph.cgraph) {
        NativeActionRegion native_region(model);
        ggml_context * ctx = graph.ctx;
        graph.prompt_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, cfg.t5_hidden, cfg.n_lang);
        ggml_set_input(graph.prompt_input);
        ggml_tensor * unused_frequency = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        ConditionOutput condition = build_condition(
            ctx, model, "action_condition_embedder", unused_frequency,
            graph.prompt_input, cfg.expert_h, 1);
        graph.action_prompt = native_bf16(model) &&
                              condition.text->type != GGML_TYPE_BF16
            ? ggml_cast(ctx, condition.text, GGML_TYPE_BF16)
            : condition.text;
        for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
            const std::string prefix = "gwp.blocks." + std::to_string(layer) +
                                       ".action_expert";
            CrossKv kv = build_cross_kv(ctx, model, prefix, graph.action_prompt);
            graph.prompt_keys.push_back(kv.k);
            graph.prompt_values.push_back(kv.v);
        }
        ggml_set_output(graph.action_prompt);
        graph.cgraph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph.cgraph, graph.action_prompt);
        for (size_t index = 0; index < graph.prompt_keys.size(); ++index) {
            ggml_set_output(graph.prompt_keys[index]);
            ggml_set_output(graph.prompt_values[index]);
            ggml_build_forward_expand(graph.cgraph, graph.prompt_keys[index]);
            ggml_build_forward_expand(graph.cgraph, graph.prompt_values[index]);
        }
        ggml_graph_assign_uid(graph.cgraph);
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend));
        if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, graph.cgraph)) {
            return false;
        }
    }
    ggml_backend_tensor_set(
        graph.prompt_input, prompt.data(), 0, prompt.size() * sizeof(float));
    if (ggml_backend_graph_compute(model.backend, graph.cgraph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    PrefixStorage & storage = *model.prefix_storage;
    ggml_backend_tensor_copy(graph.action_prompt, storage.action_prompt);
    for (size_t index = 0; index < graph.prompt_keys.size(); ++index) {
        ggml_backend_tensor_copy(graph.prompt_keys[index], storage.prompt_keys[index]);
        ggml_backend_tensor_copy(graph.prompt_values[index], storage.prompt_values[index]);
    }
    ggml_backend_synchronize(model.backend);
    model.projected_prompt_signature = prompt;
    model.stats.projected_prompt_hits = model.projected_prompt_hits;
    model.stats.projected_prompt_misses = model.projected_prompt_misses;
    model.stats.ms_prompt_projection = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    return true;
}

bool build_prefix_cache(Gwp05ModelArch & model,
                        const std::vector<float> & state,
                        const std::vector<float> & reference,
                        const std::vector<float> & prompt) {
    NativeActionRegion native_region(model);
    const Config & cfg = model.cfg;
    const int64_t visual_tokens = cfg.n_img;
    const int64_t prefix_tokens = visual_tokens + 1;
    const int64_t latent_width = cfg.image_width / 16;
    const int64_t latent_height = cfg.image_height / 16;
    const int64_t grid_width = latent_width / 2;
    const int64_t grid_height = latent_height / 2;
    auto require_native_bf16 = [&](const char * name, ggml_tensor * tensor) {
        if (!native_bf16(model) || (tensor && tensor->type == GGML_TYPE_BF16)) {
            return true;
        }
        std::fprintf(stderr,
                     "wam(gwp05): native prefix tensor %s is %s, expected BF16\n",
                     name, tensor ? ggml_type_name(tensor->type) : "null");
        return false;
    };

    if (state.size() != static_cast<size_t>(cfg.max_state_dim) ||
        reference.size() != static_cast<size_t>(latent_width * latent_height * cfg.vae_z_dim) ||
        prompt.size() != static_cast<size_t>(cfg.n_lang * cfg.t5_hidden) ||
        prefix_tokens != cfg.n_prefix) {
        std::fprintf(stderr, "wam(gwp05): prefix inputs do not match configured structural shapes\n");
        return false;
    }
    if (!ensure_prefix_storage(model) ||
        !update_projected_prompt_cache(model, prompt)) {
        return false;
    }
    const bool persistent_prompt = cross_request_prompt_cache_enabled(model);

    if (!model.prefix_graph) model.prefix_graph = std::make_unique<PrefixGraph>();
    PrefixGraph & graph = *model.prefix_graph;
    if (!graph.ctx) {
        ggml_init_params params{};
        params.mem_size = 256u * 1024u * 1024u;
        params.no_alloc = true;
        graph.ctx = ggml_init(params);
        if (!graph.ctx) return false;
    }
    ggml_context * ctx = graph.ctx;
    if (!graph.cgraph) {
        const auto graph_build_begin = std::chrono::steady_clock::now();
        graph.state_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_state_dim, 1);
        graph.reference_input = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, latent_width, latent_height, cfg.vae_z_dim, 1);
        graph.prompt_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.t5_hidden, cfg.n_lang);
        graph.action_frequency = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        graph.visual_frequency = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, visual_tokens);
        graph.state_position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        graph.visual_time = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
        graph.visual_height = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
        graph.visual_width = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
        for (ggml_tensor * input : {graph.state_input, graph.reference_input, graph.prompt_input,
                                    graph.action_frequency, graph.visual_frequency,
                                    graph.state_position, graph.visual_time,
                                    graph.visual_height, graph.visual_width}) ggml_set_input(input);

        ggml_tensor * state_hidden = action_encoder(
            ctx, model, "state_encoder", graph.state_input);
        ggml_tensor * visual_hidden = ggml_conv_2d(
            ctx, model.weight("gwp.patch_embedding.weight"), graph.reference_input,
            2, 2, 0, 0, 1, 1);
        visual_hidden = ggml_add(
            ctx, visual_hidden, ggml_reshape_4d(
                ctx, ggml_cast(ctx, model.weight("gwp.patch_embedding.bias"),
                               visual_hidden->type),
                1, 1, cfg.hidden, 1));
        if (native_action_region(model)) {
            if (visual_hidden->type != GGML_TYPE_BF16) {
                visual_hidden = ggml_cast(ctx, visual_hidden, GGML_TYPE_BF16);
            }
        }
        visual_hidden = ggml_reshape_2d(
            ctx, ggml_cont(ctx, ggml_permute(ctx, visual_hidden, 1, 2, 0, 3)),
            cfg.hidden, visual_tokens);
        if (!require_native_bf16("state_hidden_in", state_hidden) ||
            !require_native_bf16("visual_hidden_in", visual_hidden)) {
            return false;
        }
        if (cache_debug_dump_enabled()) {
            graph.state_hidden_in = ggml_dup(ctx, state_hidden);
            graph.visual_hidden_in = ggml_dup(ctx, visual_hidden);
        }
        ConditionOutput state_condition = build_condition(
            ctx, model, "action_condition_embedder", graph.action_frequency,
            graph.prompt_input, cfg.expert_h, 1,
            persistent_prompt ? model.prefix_storage->action_prompt : nullptr);
        // The text branch depends only on prompt_input and fixed text-embedder weights.
        if (action_prompt_cache_enabled() && !persistent_prompt) {
            graph.action_prompt = native_bf16(model) &&
                                  state_condition.text->type != GGML_TYPE_BF16
                ? ggml_cast(ctx, state_condition.text, GGML_TYPE_BF16)
                : state_condition.text;
        }
        ConditionOutput visual_condition = build_condition(
            ctx, model, "condition_embedder", graph.visual_frequency,
            graph.prompt_input, cfg.hidden, visual_tokens);

        for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
            const std::string block = "gwp.blocks." + std::to_string(layer);
            const std::string action_prefix = block + ".action_expert";
            const std::string visual_prefix = block + ".visual_expert";
            ggml_tensor * at = model.weight(action_prefix + ".scale_shift_table");
            ggml_tensor * vt = model.weight(visual_prefix + ".scale_shift_table");
            ggml_tensor * am = modulation_add(ctx, model, state_condition.modulation, at);
            ggml_tensor * vm = modulation_add(ctx, model, visual_condition.modulation, vt);
            ggml_tensor * as = modulation_part(ctx, am, 0, cfg.expert_h, 1);
            ggml_tensor * az = modulation_part(ctx, am, 1, cfg.expert_h, 1);
            ggml_tensor * ag = modulation_part(ctx, am, 2, cfg.expert_h, 1);
            ggml_tensor * acs = modulation_part(ctx, am, 3, cfg.expert_h, 1);
            ggml_tensor * acz = modulation_part(ctx, am, 4, cfg.expert_h, 1);
            ggml_tensor * acg = modulation_part(ctx, am, 5, cfg.expert_h, 1);
            ggml_tensor * vs = modulation_part(ctx, vm, 0, cfg.hidden, visual_tokens);
            ggml_tensor * vz = modulation_part(ctx, vm, 1, cfg.hidden, visual_tokens);
            ggml_tensor * vg = modulation_part(ctx, vm, 2, cfg.hidden, visual_tokens);
            ggml_tensor * vcs = modulation_part(ctx, vm, 3, cfg.hidden, visual_tokens);
            ggml_tensor * vcz = modulation_part(ctx, vm, 4, cfg.hidden, visual_tokens);
            ggml_tensor * vcg = modulation_part(ctx, vm, 5, cfg.hidden, visual_tokens);
            ExpertQkv sqkv = expert_qkv(ctx, model, action_prefix, state_hidden, az, as,
                                        graph.state_position, graph.visual_time,
                                        graph.visual_height, graph.visual_width, false);
            ExpertQkv vqkv = expert_qkv(ctx, model, visual_prefix, visual_hidden, vz, vs,
                                        graph.state_position, graph.visual_time,
                                        graph.visual_height, graph.visual_width, true);
            ggml_tensor * q = ggml_concat(ctx, sqkv.q, vqkv.q, 2);
            ggml_tensor * k = ggml_concat(ctx, sqkv.k, vqkv.k, 2);
            ggml_tensor * v = ggml_concat(ctx, sqkv.v, vqkv.v, 2);
            ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
            const float attention_scale =
                1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
            ggml_tensor * attended = native_attention_context(
                ctx, model, Q, K, V, attention_scale);
            if (!attended) {
                ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                ggml_tensor * probs = ggml_soft_max_ext(
                    ctx, scores, nullptr, attention_scale, 0.0f);
                attended = ggml_reshape_2d(
                    ctx, ggml_cont(ctx, ggml_permute(
                        ctx, ggml_mul_mat(ctx, V, probs), 0, 2, 1, 3)),
                    cfg.hidden, prefix_tokens);
            }
            ggml_tensor * state_attended = ggml_view_2d(ctx, attended, cfg.hidden, 1,
                                                        attended->nb[1], 0);
            ggml_tensor * visual_attended = ggml_view_2d(
                ctx, attended, cfg.hidden, visual_tokens, attended->nb[1], attended->nb[1]);
            const bool use_bf16_value_output = bf16_attention_value_output(model);
            state_attended = linear_with_activation(
                ctx, model.weight(action_prefix + ".attn1.to_out.0.weight"),
                model.weight(action_prefix + ".attn1.to_out.0.bias"),
                state_attended, use_bf16_value_output);
            visual_attended = linear_with_activation(
                ctx, model.weight(visual_prefix + ".attn1.to_out.0.weight"),
                model.weight(visual_prefix + ".attn1.to_out.0.bias"),
                visual_attended, use_bf16_value_output);
            state_hidden = gated_residual(ctx, model, state_hidden, state_attended, ag);
            visual_hidden = gated_residual(ctx, model, visual_hidden, visual_attended, vg);
            CrossKv action_prompt_kv;
            if (persistent_prompt) {
                action_prompt_kv.k = model.prefix_storage->prompt_keys[layer];
                action_prompt_kv.v = model.prefix_storage->prompt_values[layer];
            } else if (prompt_kv_cache_enabled()) {
                action_prompt_kv = build_cross_kv(
                    ctx, model, action_prefix, state_condition.text);
            }
            state_hidden = expert_cross_ffn(ctx, model, action_prefix, state_hidden,
                                            state_condition.text, acs, acz, acg,
                                            action_prompt_kv.k, action_prompt_kv.v);
            visual_hidden = expert_cross_ffn(ctx, model, visual_prefix, visual_hidden,
                                             visual_condition.text, vcs, vcz, vcg);
            if (bf16_kv_cache_storage(model)) {
                if (K->type != GGML_TYPE_BF16) K = ggml_cast(ctx, K, GGML_TYPE_BF16);
                if (V->type != GGML_TYPE_BF16) V = ggml_cast(ctx, V, GGML_TYPE_BF16);
                if (action_prompt_kv.k && action_prompt_kv.k->type != GGML_TYPE_BF16) {
                    action_prompt_kv.k = ggml_cast(ctx, action_prompt_kv.k, GGML_TYPE_BF16);
                }
                if (action_prompt_kv.v && action_prompt_kv.v->type != GGML_TYPE_BF16) {
                    action_prompt_kv.v = ggml_cast(ctx, action_prompt_kv.v, GGML_TYPE_BF16);
                }
            }
            if (!require_native_bf16("state_hidden_out", state_hidden) ||
                !require_native_bf16("visual_hidden_out", visual_hidden) ||
                !require_native_bf16("prefix_key", K) ||
                !require_native_bf16("prefix_value", V) ||
                (action_prompt_kv.k &&
                 (!require_native_bf16("prompt_key", action_prompt_kv.k) ||
                  !require_native_bf16("prompt_value", action_prompt_kv.v)))) {
                return false;
            }
            graph.keys.push_back(K);
            graph.values.push_back(V);
            if (action_prompt_kv.k && !persistent_prompt) {
                graph.prompt_keys.push_back(action_prompt_kv.k);
                graph.prompt_values.push_back(action_prompt_kv.v);
            }
            if (cache_debug_dump_enabled()) {
                graph.state_hidden_out.push_back(ggml_dup(ctx, state_hidden));
                graph.visual_hidden_out.push_back(ggml_dup(ctx, visual_hidden));
            }
        }
        graph.cgraph = ggml_new_graph_custom(ctx, 32768, false);
        for (size_t i = 0; i < graph.keys.size(); ++i) {
            ggml_set_output(graph.keys[i]);
            ggml_set_output(graph.values[i]);
            ggml_build_forward_expand(graph.cgraph, graph.keys[i]);
            ggml_build_forward_expand(graph.cgraph, graph.values[i]);
        }
        if (graph.action_prompt) {
            ggml_set_output(graph.action_prompt);
            ggml_build_forward_expand(graph.cgraph, graph.action_prompt);
        }
        for (size_t i = 0; i < graph.prompt_keys.size(); ++i) {
            ggml_set_output(graph.prompt_keys[i]);
            ggml_set_output(graph.prompt_values[i]);
            ggml_build_forward_expand(graph.cgraph, graph.prompt_keys[i]);
            ggml_build_forward_expand(graph.cgraph, graph.prompt_values[i]);
        }
        if (cache_debug_dump_enabled()) {
            for (ggml_tensor * tensor : {graph.state_hidden_in, graph.visual_hidden_in}) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph.cgraph, tensor);
            }
            for (size_t i = 0; i < graph.state_hidden_out.size(); ++i) {
                ggml_set_output(graph.state_hidden_out[i]);
                ggml_set_output(graph.visual_hidden_out[i]);
                ggml_build_forward_expand(graph.cgraph, graph.state_hidden_out[i]);
                ggml_build_forward_expand(graph.cgraph, graph.visual_hidden_out[i]);
            }
        }
        ggml_graph_assign_uid(graph.cgraph);
        graph.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, graph.cgraph)) return false;
        audit_mixed_binary_nodes("prefix", graph.cgraph);

        PrefixStorage & storage = *model.prefix_storage;
        if (graph.keys.size() != static_cast<size_t>(cfg.n_layers) ||
            graph.values.size() != static_cast<size_t>(cfg.n_layers) ||
            storage.keys.size() != static_cast<size_t>(cfg.n_layers) ||
            storage.values.size() != static_cast<size_t>(cfg.n_layers) ||
            (prompt_kv_cache_enabled() &&
             ((!persistent_prompt &&
               (graph.prompt_keys.size() != static_cast<size_t>(cfg.n_layers) ||
                graph.prompt_values.size() != static_cast<size_t>(cfg.n_layers))) ||
              storage.prompt_keys.size() != static_cast<size_t>(cfg.n_layers) ||
              storage.prompt_values.size() != static_cast<size_t>(cfg.n_layers)))) {
            std::fprintf(stderr, "wam(gwp05): prefix cache layer count does not match model layers\n");
            return false;
        }
        if (storage.action_prompt &&
            !tensor_shape_is(storage.action_prompt, "projected action prompt",
                             {cfg.expert_h, cfg.n_lang})) {
            return false;
        }
        for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
            const size_t index = static_cast<size_t>(layer);
            if (!tensor_shape_is(storage.keys[index], "prefix key",
                                 {cfg.head_dim, cfg.n_prefix, cfg.n_q_heads}) ||
                !tensor_shape_is(storage.values[index], "prefix value",
                                 {cfg.n_prefix, cfg.head_dim, cfg.n_q_heads})) {
                return false;
            }
            if (prompt_kv_cache_enabled() &&
                (!tensor_shape_is(storage.prompt_keys[index], "prompt key",
                                  {cfg.head_dim, cfg.n_lang, cfg.n_q_heads}) ||
                 !tensor_shape_is(storage.prompt_values[index], "prompt value",
                                  {cfg.n_lang, cfg.head_dim, cfg.n_q_heads}))) {
                return false;
            }
        }
        model.stats.ms_prefix_graph_build = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - graph_build_begin).count();
    }

    const std::vector<float> zero = timestep_embedding(0.0f, 256);
    std::vector<float> visual_frequency(static_cast<size_t>(256) * visual_tokens);
    for (int64_t i = 0; i < visual_tokens; ++i)
        std::copy(zero.begin(), zero.end(), visual_frequency.begin() + i * 256);
    const auto visual_positions = gwp05_semantics::visual_positions(grid_height, grid_width);
    const int32_t state_pos = 0;
    auto set_input = [](ggml_tensor * tensor, const void * data, size_t size) {
        if (tensor->buffer) ggml_backend_tensor_set(tensor, data, 0, size);
    };
    set_input(graph.state_input, state.data(), state.size() * sizeof(float));
    set_input(graph.reference_input, reference.data(), reference.size() * sizeof(float));
    set_input(graph.prompt_input, prompt.data(), prompt.size() * sizeof(float));
    set_input(graph.action_frequency, zero.data(), zero.size() * sizeof(float));
    set_input(graph.visual_frequency, visual_frequency.data(), visual_frequency.size() * sizeof(float));
    set_input(graph.state_position, &state_pos, sizeof(state_pos));
    set_input(graph.visual_time, visual_positions.time.data(),
              visual_positions.time.size() * sizeof(int32_t));
    set_input(graph.visual_height, visual_positions.height.data(),
              visual_positions.height.size() * sizeof(int32_t));
    set_input(graph.visual_width, visual_positions.width.data(),
              visual_positions.width.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(model.backend, graph.cgraph) != GGML_STATUS_SUCCESS) return false;
    if (cache_debug_dump_enabled()) {
        debug_dump_tensor("prefix_state_hidden_in", graph.state_hidden_in);
        debug_dump_tensor("prefix_visual_hidden_in", graph.visual_hidden_in);
        debug_dump_tensor("prefix_projected_action_prompt", graph.action_prompt);
        for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
            char name[96];
            std::snprintf(name, sizeof(name), "prefix_layer_%02lld_state_hidden_out",
                          static_cast<long long>(layer));
            debug_dump_tensor(name, graph.state_hidden_out[static_cast<size_t>(layer)]);
            std::snprintf(name, sizeof(name), "prefix_layer_%02lld_visual_hidden_out",
                          static_cast<long long>(layer));
            debug_dump_tensor(name, graph.visual_hidden_out[static_cast<size_t>(layer)]);
            std::snprintf(name, sizeof(name), "prefix_layer_%02lld_self_key",
                          static_cast<long long>(layer));
            debug_dump_tensor(name, graph.keys[static_cast<size_t>(layer)]);
            std::snprintf(name, sizeof(name), "prefix_layer_%02lld_self_value",
                          static_cast<long long>(layer));
            debug_dump_tensor(name, graph.values[static_cast<size_t>(layer)]);
            if (graph.prompt_keys.size() == static_cast<size_t>(cfg.n_layers) &&
                graph.prompt_values.size() == static_cast<size_t>(cfg.n_layers)) {
                std::snprintf(name, sizeof(name), "prefix_layer_%02lld_prompt_key",
                              static_cast<long long>(layer));
                debug_dump_tensor(name, graph.prompt_keys[static_cast<size_t>(layer)]);
                std::snprintf(name, sizeof(name), "prefix_layer_%02lld_prompt_value",
                              static_cast<long long>(layer));
                debug_dump_tensor(name, graph.prompt_values[static_cast<size_t>(layer)]);
            }
        }
    }
    for (size_t i = 0; i < graph.keys.size(); ++i) {
        ggml_backend_tensor_copy(graph.keys[i], model.prefix_storage->keys[i]);
        ggml_backend_tensor_copy(graph.values[i], model.prefix_storage->values[i]);
    }
    if (graph.action_prompt) {
        ggml_backend_tensor_copy(graph.action_prompt, model.prefix_storage->action_prompt);
    }
    for (size_t i = 0; i < graph.prompt_keys.size(); ++i) {
        ggml_backend_tensor_copy(
            graph.prompt_keys[i], model.prefix_storage->prompt_keys[i]);
        ggml_backend_tensor_copy(
            graph.prompt_values[i], model.prefix_storage->prompt_values[i]);
    }
    ggml_backend_synchronize(model.backend);
    return true;
}
