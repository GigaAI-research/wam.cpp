ggml_tensor * embodiment_weight(ggml_context * ctx, ggml_tensor * weight,
                                int64_t input_dim, int64_t output_dim,
                                int64_t embodiment) {
    ggml_tensor * selected = ggml_view_2d(
        ctx, weight, output_dim, input_dim, weight->nb[1],
        static_cast<size_t>(embodiment) * weight->nb[2]);
    return ggml_cont(ctx, ggml_transpose(ctx, selected));
}

ggml_tensor * embodiment_bias(ggml_context * ctx, ggml_tensor * bias,
                              int64_t output_dim, int64_t embodiment) {
    return ggml_view_1d(
        ctx, bias, output_dim, static_cast<size_t>(embodiment) * bias->nb[1]);
}

ggml_tensor * action_encoder(ggml_context * ctx, Gwp05ModelArch & model,
                             const std::string & name, ggml_tensor * input) {
    const Config & cfg = model.cfg;
    const std::string prefix = "gwp." + name;
    const bool use_bf16 = bf16_projection_activations(model);
    ggml_tensor * hidden = linear_with_activation(
        ctx,
        embodiment_weight(ctx, model.weight(prefix + ".in_proj.weight"),
                           cfg.max_action_dim, 128, cfg.embodiment_id),
        embodiment_bias(ctx, model.weight(prefix + ".in_proj.bias"),
                        128, cfg.embodiment_id),
        input, use_bf16);
    hidden = ggml_gelu_erf(ctx, hidden);
    hidden = linear_with_activation(ctx, model.weight(prefix + ".mid_proj.weight"),
                    model.weight(prefix + ".mid_proj.bias"), hidden, use_bf16);
    hidden = ggml_gelu_erf(ctx, hidden);
    return linear_with_activation(ctx, model.weight(prefix + ".out_proj.weight"),
                  model.weight(prefix + ".out_proj.bias"), hidden, use_bf16);
}

ggml_tensor * action_decoder(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * input) {
    const Config & cfg = model.cfg;
    const std::string prefix = "gwp.action_decoder";
    const bool use_bf16 = bf16_projection_activations(model);
    ggml_tensor * hidden = linear_with_activation(
        ctx, model.weight(prefix + ".in_proj.weight"),
        model.weight(prefix + ".in_proj.bias"), input, use_bf16);
    hidden = ggml_gelu_erf(ctx, hidden);
    hidden = linear_with_activation(ctx, model.weight(prefix + ".mid_proj.weight"),
                    model.weight(prefix + ".mid_proj.bias"), hidden, use_bf16);
    hidden = ggml_gelu_erf(ctx, hidden);
    return linear_with_activation(
        ctx,
        embodiment_weight(ctx, model.weight(prefix + ".out_proj.weight"),
                           128, cfg.max_action_dim, cfg.embodiment_id),
        embodiment_bias(ctx, model.weight(prefix + ".out_proj.bias"),
                        cfg.max_action_dim, cfg.embodiment_id),
        hidden, use_bf16);
}

struct ConditionOutput {
    ggml_tensor * temb = nullptr;
    ggml_tensor * modulation = nullptr;
    ggml_tensor * text = nullptr;
};

ConditionOutput build_condition(ggml_context * ctx, Gwp05ModelArch & model,
                                const std::string & component,
                                ggml_tensor * time_frequencies,
                                ggml_tensor * prompt,
                                int64_t hidden, int64_t tokens,
                                ggml_tensor * projected_text = nullptr,
                                bool build_text = true) {
    const std::string prefix = "gwp." + component;
    const bool use_bf16 = bf16_projection_activations(model);
    ConditionOutput output;
    output.temb = linear_with_activation(ctx, model.weight(prefix + ".time_embedder.linear_1.weight"),
                         model.weight(prefix + ".time_embedder.linear_1.bias"),
                         time_frequencies, use_bf16);
    output.temb = ggml_silu(ctx, output.temb);
    output.temb = linear_with_activation(ctx, model.weight(prefix + ".time_embedder.linear_2.weight"),
                         model.weight(prefix + ".time_embedder.linear_2.bias"), output.temb, use_bf16);
    output.modulation = linear_with_activation(ctx, model.weight(prefix + ".time_proj.weight"),
                               model.weight(prefix + ".time_proj.bias"),
                               ggml_silu(ctx, output.temb), use_bf16);
    output.modulation = ggml_reshape_3d(ctx, output.modulation, hidden, 6, tokens);
    output.text = build_text ? projected_text : nullptr;
    if (build_text && !output.text) {
        output.text = linear_with_activation(
            ctx, model.weight(prefix + ".text_embedder.linear_1.weight"),
            model.weight(prefix + ".text_embedder.linear_1.bias"), prompt, use_bf16);
        output.text = ggml_gelu(ctx, output.text);
        output.text = linear_with_activation(
            ctx, model.weight(prefix + ".text_embedder.linear_2.weight"),
            model.weight(prefix + ".text_embedder.linear_2.bias"), output.text, use_bf16);
    }
    return output;
}

struct CrossKv {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

CrossKv build_cross_kv(ggml_context * ctx, Gwp05ModelArch & model,
                       const std::string & prefix, ggml_tensor * encoder_hidden) {
    const Config & cfg = model.cfg;
    const std::string cross = prefix + ".attn2";
    const bool use_bf16_qkv = bf16_qkv_activations(model);
    CrossKv output;
    output.k = linear_with_activation(
        ctx, model.weight(cross + ".to_k.weight"),
        model.weight(cross + ".to_k.bias"), encoder_hidden, use_bf16_qkv);
    output.v = linear_with_activation(
        ctx, model.weight(cross + ".to_v.weight"),
        model.weight(cross + ".to_v.bias"), encoder_hidden, use_bf16_qkv);
    output.k = rms_norm(
        ctx, output.k, model.weight(cross + ".norm_k.weight"), cfg.norm_eps);
    output.k = ggml_reshape_3d(
        ctx, output.k, cfg.head_dim, cfg.n_q_heads, cfg.n_lang);
    output.v = ggml_reshape_3d(
        ctx, output.v, cfg.head_dim, cfg.n_q_heads, cfg.n_lang);
    output.k = ggml_cont(ctx, ggml_permute(ctx, output.k, 0, 2, 1, 3));
    output.v = ggml_cont(ctx, ggml_permute(ctx, output.v, 1, 2, 0, 3));
    if (native_action_region(model) && output.k->type != GGML_TYPE_BF16) {
        output.k = ggml_cast(ctx, output.k, GGML_TYPE_BF16);
    }
    if (bf16_attention_value_output(model) && output.v->type != GGML_TYPE_BF16) {
        output.v = ggml_cast(ctx, output.v, GGML_TYPE_BF16);
    }
    return output;
}

ggml_tensor * modulation_part(ggml_context * ctx, ggml_tensor * combined,
                              int index, int64_t hidden, int64_t tokens) {
    return ggml_view_2d(
        ctx, combined, hidden, tokens, combined->nb[2],
        static_cast<size_t>(index) * combined->nb[1]);
}

ggml_tensor * modulation_add(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * a, ggml_tensor * b) {
    const ggml_type target = unsafe_fast_native_bf16(model)
        ? GGML_TYPE_BF16
        : (native_action_region(model) ? GGML_TYPE_F32 : a->type);
    if (a->type != target) a = ggml_cast(ctx, a, target);
    if (b->type != target) b = ggml_cast(ctx, b, target);
    return ggml_add(ctx, a, b);
}

ggml_tensor * modulated_norm(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * scale, ggml_tensor * shift,
                             float eps) {
    if (input->type == GGML_TYPE_BF16 && scale->type == shift->type &&
        (scale->type == GGML_TYPE_F32 || scale->type == GGML_TYPE_BF16)) {
        return ggml_norm_modulation_bf16(ctx, input, scale, shift, eps);
    }
    if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    if (scale->type != GGML_TYPE_F32) scale = ggml_cast(ctx, scale, GGML_TYPE_F32);
    if (shift->type != GGML_TYPE_F32) shift = ggml_cast(ctx, shift, GGML_TYPE_F32);
    ggml_tensor * normalized = ggml_norm(ctx, input, eps);
    return ggml_add(ctx, ggml_add(ctx, normalized, ggml_mul(ctx, normalized, scale)), shift);
}

ggml_tensor * gated_residual(ggml_context * ctx, Gwp05ModelArch & model,
                             ggml_tensor * hidden, ggml_tensor * branch,
                             ggml_tensor * gate) {
    if (!native_action_region(model)) {
        return ggml_add(ctx, hidden, ggml_mul(ctx, branch, gate));
    }
    if (hidden->type != GGML_TYPE_BF16) hidden = ggml_cast(ctx, hidden, GGML_TYPE_BF16);
    if (branch->type != GGML_TYPE_BF16) branch = ggml_cast(ctx, branch, GGML_TYPE_BF16);
    const ggml_type gate_type = unsafe_fast_native_bf16(model)
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    if (gate->type != gate_type) gate = ggml_cast(ctx, gate, gate_type);
    return ggml_gated_residual_bf16(ctx, hidden, branch, gate);
}

ggml_tensor * apply_rope_1d(ggml_context * ctx, ggml_tensor * input,
                            ggml_tensor * positions, int64_t head_dim) {
    return ggml_rope_ext(ctx, input, positions, nullptr,
                         static_cast<int>(head_dim), GGML_ROPE_TYPE_NORMAL, 0,
                         10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

ggml_tensor * apply_visual_rope(ggml_context * ctx, ggml_tensor * input,
                                ggml_tensor * time_positions,
                                ggml_tensor * height_positions,
                                ggml_tensor * width_positions,
                                int64_t heads, int64_t tokens) {
    ggml_tensor * result = nullptr;
    size_t offset = 0;
    ggml_tensor * positions[3] = {time_positions, height_positions, width_positions};
    for (int section = 0; section < 3; ++section) {
        ggml_tensor * view = ggml_view_3d(
            ctx, input, gwp05_semantics::kVisualRopeSections[section], heads, tokens,
            input->nb[1], input->nb[2], offset);
        ggml_tensor * rotated = ggml_rope_ext(
            ctx, view, positions[section], nullptr,
            gwp05_semantics::kVisualRopeSections[section],
            GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        result = result ? ggml_concat(ctx, result, rotated, 0) : rotated;
        offset += static_cast<size_t>(gwp05_semantics::kVisualRopeSections[section]) * input->nb[0];
    }
    return ggml_cont(ctx, result);
}

struct ExpertQkv {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

ExpertQkv expert_qkv(ggml_context * ctx, Gwp05ModelArch & model,
                     const std::string & prefix, ggml_tensor * input,
                     ggml_tensor * scale, ggml_tensor * shift,
                     ggml_tensor * action_positions,
                     ggml_tensor * visual_time,
                     ggml_tensor * visual_height,
                     ggml_tensor * visual_width,
                     bool visual) {
    const Config & cfg = model.cfg;
    const int64_t tokens = input->ne[1];
    ggml_tensor * normalized = modulated_norm(ctx, input, scale, shift, cfg.norm_eps);
    const bool use_bf16_qkv = bf16_qkv_activations(model);
    const std::string attention = prefix + ".attn1";
    ExpertQkv output;
    if (packed_self_qkv(model)) {
        ggml_tensor * qkv = linear_with_activation(
            ctx, model.weight(attention + ".to_qkv.weight"),
            model.weight(attention + ".to_qkv.bias"), normalized, use_bf16_qkv);
        const size_t element_size = ggml_type_size(qkv->type);
        output.q = ggml_view_2d(ctx, qkv, cfg.hidden, tokens, qkv->nb[1], 0);
        output.k = ggml_view_2d(
            ctx, qkv, cfg.hidden, tokens, qkv->nb[1], cfg.hidden * element_size);
        output.v = ggml_view_2d(
            ctx, qkv, cfg.hidden, tokens, qkv->nb[1], 2 * cfg.hidden * element_size);
        output.q = ggml_cont(ctx, output.q);
        output.k = ggml_cont(ctx, output.k);
        output.v = ggml_cont(ctx, output.v);
    } else {
        output.q = linear_with_activation(ctx, model.weight(attention + ".to_q.weight"),
                          model.weight(attention + ".to_q.bias"), normalized, use_bf16_qkv);
        output.k = linear_with_activation(ctx, model.weight(attention + ".to_k.weight"),
                          model.weight(attention + ".to_k.bias"), normalized, use_bf16_qkv);
        output.v = linear_with_activation(ctx, model.weight(attention + ".to_v.weight"),
                          model.weight(attention + ".to_v.bias"), normalized, use_bf16_qkv);
    }
    output.q = rms_norm(ctx, output.q, model.weight(attention + ".norm_q.weight"), cfg.norm_eps);
    output.k = rms_norm(ctx, output.k, model.weight(attention + ".norm_k.weight"), cfg.norm_eps);
    output.q = ggml_reshape_3d(ctx, output.q, cfg.head_dim, cfg.n_q_heads, tokens);
    output.k = ggml_reshape_3d(ctx, output.k, cfg.head_dim, cfg.n_q_heads, tokens);
    output.v = ggml_reshape_3d(ctx, output.v, cfg.head_dim, cfg.n_q_heads, tokens);
    if (bf16_attention_value_output(model) && output.v->type != GGML_TYPE_BF16) {
        output.v = ggml_cast(ctx, output.v, GGML_TYPE_BF16);
    }
    if (visual) {
        output.q = apply_visual_rope(ctx, output.q, visual_time, visual_height,
                                     visual_width, cfg.n_q_heads, tokens);
        output.k = apply_visual_rope(ctx, output.k, visual_time, visual_height,
                                     visual_width, cfg.n_q_heads, tokens);
    } else {
        output.q = apply_rope_1d(ctx, output.q, action_positions, cfg.head_dim);
        output.k = apply_rope_1d(ctx, output.k, action_positions, cfg.head_dim);
    }
    if (native_action_region(model)) {
        output.q = ggml_cast(ctx, output.q, GGML_TYPE_BF16);
        output.k = ggml_cast(ctx, output.k, GGML_TYPE_BF16);
    }
    return output;
}

ggml_tensor * native_attention_context(ggml_context * ctx,
                                       Gwp05ModelArch & model,
                                       ggml_tensor * q,
                                       ggml_tensor * k,
                                       ggml_tensor * transposed_v,
                                       float scale) {
    (void) ctx;
    (void) model;
    (void) q;
    (void) k;
    (void) transposed_v;
    (void) scale;
    return nullptr;
}

ggml_tensor * expert_cross_ffn(ggml_context * ctx, Gwp05ModelArch & model,
                               const std::string & prefix,
                               ggml_tensor * hidden,
                               ggml_tensor * encoder_hidden,
                               ggml_tensor * shift,
                               ggml_tensor * scale,
                               ggml_tensor * gate,
                               ggml_tensor * cached_k = nullptr,
                               ggml_tensor * cached_v = nullptr) {
    const Config & cfg = model.cfg;
    const std::string cross = prefix + ".attn2";
    ggml_tensor * normalized = layer_norm(
        ctx, hidden, model.weight(prefix + ".norm2.weight"),
        model.weight(prefix + ".norm2.bias"), cfg.norm_eps);
    ggml_tensor * attended = nullptr;
    if (cached_k && cached_v) {
        const int64_t query_tokens = normalized->ne[1];
        ggml_tensor * q = linear_with_activation(
            ctx, model.weight(cross + ".to_q.weight"),
            model.weight(cross + ".to_q.bias"), normalized, bf16_qkv_activations(model));
        q = rms_norm(ctx, q, model.weight(cross + ".norm_q.weight"), cfg.norm_eps);
        q = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, query_tokens);
        if (native_action_region(model)) q = ggml_cast(ctx, q, GGML_TYPE_BF16);
        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        const float attention_scale =
            1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
        attended = native_attention_context(
            ctx, model, Q, cached_k, cached_v, attention_scale);
        if (!attended) {
            ggml_tensor * scores = ggml_mul_mat(ctx, cached_k, Q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            ggml_tensor * probabilities = ggml_soft_max_ext(
                ctx, scores, nullptr, attention_scale, 0.0f);
            attended = ggml_mul_mat(ctx, cached_v, probabilities);
            attended = ggml_reshape_2d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)),
                cfg.hidden, query_tokens);
        }
        attended = linear_with_activation(
            ctx, model.weight(cross + ".to_out.0.weight"),
            model.weight(cross + ".to_out.0.bias"), attended,
            bf16_attention_value_output(model));
    } else {
        attended = build_attention(
            ctx, normalized, encoder_hidden,
            model.weight(cross + ".to_q.weight"), model.weight(cross + ".to_q.bias"),
            model.weight(cross + ".to_k.weight"), model.weight(cross + ".to_k.bias"),
            model.weight(cross + ".to_v.weight"), model.weight(cross + ".to_v.bias"),
            model.weight(cross + ".to_out.0.weight"), model.weight(cross + ".to_out.0.bias"),
            model.weight(cross + ".norm_q.weight"), model.weight(cross + ".norm_k.weight"),
            cfg.n_q_heads, cfg.head_dim, nullptr,
            1.0f / std::sqrt(static_cast<float>(cfg.head_dim)), cfg.norm_eps,
            bf16_qkv_activations(model), bf16_attention_value_output(model));
    }
    hidden = ggml_add(ctx, hidden, attended);
    normalized = modulated_norm(ctx, hidden, scale, shift, cfg.norm_eps);
    const bool use_bf16 = bf16_ffn_activations(model);
    ggml_tensor * ffn = linear_gelu_with_activation(
        ctx, model.weight(prefix + ".ffn.net.0.proj.weight"),
        model.weight(prefix + ".ffn.net.0.proj.bias"), normalized, use_bf16);
    ffn = linear_with_activation(ctx, model.weight(prefix + ".ffn.net.2.weight"),
                 model.weight(prefix + ".ffn.net.2.bias"), ffn, use_bf16);
    return gated_residual(ctx, model, hidden, ffn, gate);
}

#include "cache.cpp"

struct CachedActionBody {
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_tensor * block0_action = nullptr;
    ggml_tensor * action_condition = nullptr;
};

CachedActionBody build_cached_action_body(
        ggml_context * ctx, Gwp05ModelArch & model, ggml_tensor * action_input,
        ggml_tensor * frequency_input, ggml_tensor * positions, ggml_tensor * dt_input,
        ggml_tensor * prompt_input, bool keep_debug) {
    NativeActionRegion native_region(model);
    const Config & cfg = model.cfg;
    const int64_t tokens = cfg.action_chunk;
    const int64_t condition_tokens = frequency_input->ne[1];
    const bool cache_action_prompt = action_prompt_cache_enabled();
    const bool cache_prompt_kv = prompt_kv_cache_enabled();
    CachedActionBody output;

    ggml_tensor * hidden = action_encoder(ctx, model, "action_encoder", action_input);
    ConditionOutput condition = build_condition(
        ctx, model, "action_condition_embedder", frequency_input,
        prompt_input, cfg.expert_h, condition_tokens,
        cache_action_prompt ? model.prefix_storage->action_prompt : nullptr,
        !cache_prompt_kv);
    if (keep_debug) output.action_condition = ggml_dup(ctx, condition.modulation);
    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        const std::string prefix =
            "gwp.blocks." + std::to_string(layer) + ".action_expert";
        ggml_tensor * combined = modulation_add(
            ctx, model, condition.modulation, model.weight(prefix + ".scale_shift_table"));
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
            ctx, model, prefix, hidden, scale, shift, positions,
            nullptr, nullptr, nullptr, false);
        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, qkv.q, 0, 2, 1, 3));
        ggml_tensor * action_k = ggml_cont(
            ctx, ggml_permute(ctx, qkv.k, 0, 2, 1, 3));
        ggml_tensor * action_v = ggml_cont(
            ctx, ggml_permute(ctx, qkv.v, 1, 2, 0, 3));
        // Match complete MoT's [state, action, reference] reduction order while
        // retaining physical prefix storage as [state, reference].
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
            ggml_tensor * joint_attended = ggml_mul_mat(ctx, joint_v, probs);
            attended = ggml_reshape_2d(
                ctx, ggml_cont(ctx, ggml_permute(
                    ctx, joint_attended, 0, 2, 1, 3)),
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
        if (layer == 0 && keep_debug) output.block0_action = ggml_dup(ctx, hidden);
    }
    ggml_tensor * temb = ggml_reshape_3d(
        ctx, condition.temb, cfg.expert_h, 1, condition_tokens);
    ggml_tensor * modulation_shape = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, cfg.expert_h, 2, condition_tokens);
    ggml_tensor * action_scale_shift = model.weight("gwp.action_scale_shift_table");
    if (native_action_region(model)) {
        action_scale_shift = ggml_cast(ctx, action_scale_shift, GGML_TYPE_F32);
    }
    ggml_tensor * table = condition_tokens == 1
        ? ggml_reshape_3d(
              ctx, action_scale_shift, cfg.expert_h, 2, 1)
        : ggml_repeat(
              ctx, action_scale_shift, modulation_shape);
    ggml_tensor * modulation = modulation_add(ctx, model, table, temb);
    ggml_tensor * shift = ggml_view_2d(
        ctx, modulation, cfg.expert_h, condition_tokens, modulation->nb[2], 0);
    ggml_tensor * scale = ggml_view_2d(
        ctx, modulation, cfg.expert_h, condition_tokens,
        modulation->nb[2], modulation->nb[1]);
    ggml_tensor * normalized = modulated_norm(
        ctx, hidden, scale, shift, cfg.norm_eps);
    output.prediction = action_decoder(ctx, model, normalized);
    if (!native_action_region(model)) {
        output.prediction = ggml_cast(ctx, output.prediction, GGML_TYPE_F32);
    }
    output.action_output = scheduler_step(
        ctx, model, action_input, output.prediction, dt_input);
    return output;
}

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

bool run_mot_step(Gwp05ModelArch & model,
                  const std::vector<float> & state,
                  const std::vector<float> & action,
                  const std::vector<float> & reference,
                  const std::vector<float> & prompt,
                  float timestep,
                  float dt,
                  bool upload_action,
                  bool dump_intermediates,
                  std::vector<float> * prediction_output,
                  std::vector<float> * action_output_host) {
    NativeActionRegion native_region(model);
    const Config & cfg = model.cfg;
    const int64_t action_tokens = cfg.action_chunk + 1;
    const int64_t visual_tokens = cfg.n_img;
    const int64_t total_tokens = action_tokens + visual_tokens;
    const int64_t latent_width = cfg.image_width / 16;
    const int64_t latent_height = cfg.image_height / 16;
    const int64_t grid_width = latent_width / 2;
    const int64_t grid_height = latent_height / 2;

    if (!model.mot_graph) model.mot_graph = std::make_unique<MotGraph>();
    MotGraph & graph = *model.mot_graph;
    const bool built_graph = graph.cgraph == nullptr;
    if (!graph.ctx) {
      ggml_init_params params{};
      params.mem_size = 256u * 1024u * 1024u;
      params.no_alloc = true;
      graph.ctx = ggml_init(params);
      if (!graph.ctx) return false;
    }
    ggml_context * ctx = graph.ctx;
    if (!graph.cgraph) {
    graph.state_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, cfg.max_state_dim, 1);
    graph.action_input = ggml_new_tensor_2d(
        ctx, native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32,
        cfg.max_action_dim, cfg.action_chunk);
    const size_t action_input_bytes = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(model.backend), graph.action_input);
    graph.action_input_buffer = ggml_backend_alloc_buffer(model.backend, action_input_bytes);
    if (!graph.action_input_buffer ||
        ggml_backend_tensor_alloc(
            graph.action_input_buffer, graph.action_input,
            ggml_backend_buffer_get_base(graph.action_input_buffer)) != GGML_STATUS_SUCCESS) {
        return false;
    }
    graph.reference_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, latent_width, latent_height, cfg.vae_z_dim, 1);
    graph.prompt_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, cfg.t5_hidden, cfg.n_lang);
    for (ggml_tensor * input : {graph.state_input, graph.action_input,
                                graph.reference_input, graph.prompt_input}) {
        ggml_set_input(input);
    }

    graph.action_frequency = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 256, action_tokens);
    graph.visual_frequency = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 256, visual_tokens);
    graph.action_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, action_tokens);
    graph.visual_time = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
    graph.visual_height = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
    graph.visual_width = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, visual_tokens);
    graph.attention_mask = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, total_tokens, total_tokens, cfg.n_q_heads);
    graph.dt_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    for (ggml_tensor * input : {graph.action_frequency, graph.visual_frequency,
                                graph.action_positions, graph.visual_time,
                                graph.visual_height, graph.visual_width,
                                graph.attention_mask, graph.dt_input}) {
        ggml_set_input(input);
    }

    ggml_tensor * state_input = graph.state_input;
    ggml_tensor * action_input = graph.action_input;
    ggml_tensor * reference_input = graph.reference_input;
    ggml_tensor * prompt_input = graph.prompt_input;
    ggml_tensor * action_frequency = graph.action_frequency;
    ggml_tensor * visual_frequency = graph.visual_frequency;
    ggml_tensor * action_positions = graph.action_positions;
    ggml_tensor * visual_time = graph.visual_time;
    ggml_tensor * visual_height = graph.visual_height;
    ggml_tensor * visual_width = graph.visual_width;
    ggml_tensor * attention_mask = graph.attention_mask;

    ggml_tensor * action_hidden = action_encoder(ctx, model, "state_encoder", state_input);
    ggml_tensor * encoded_action = action_encoder(ctx, model, "action_encoder", action_input);
    action_hidden = ggml_concat(ctx, action_hidden, encoded_action, 1);
    graph.action_tokens_debug = ggml_dup(ctx, action_hidden);
    ggml_tensor * action_tokens_debug = graph.action_tokens_debug;
    ggml_tensor * visual_hidden = ggml_conv_2d(
        ctx, model.weight("gwp.patch_embedding.weight"), reference_input,
        2, 2, 0, 0, 1, 1);
    visual_hidden = ggml_add(
        ctx, visual_hidden,
        ggml_reshape_4d(ctx, ggml_cast(
                            ctx, model.weight("gwp.patch_embedding.bias"),
                            visual_hidden->type),
                        1, 1, cfg.hidden, 1));
    visual_hidden = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, visual_hidden, 1, 2, 0, 3)),
        cfg.hidden, visual_tokens);
    graph.visual_tokens_debug = ggml_dup(ctx, visual_hidden);
    ggml_tensor * visual_tokens_debug = graph.visual_tokens_debug;

    ConditionOutput action_condition = build_condition(
        ctx, model, "action_condition_embedder", action_frequency,
        prompt_input, cfg.expert_h, action_tokens);
    ConditionOutput visual_condition = build_condition(
        ctx, model, "condition_embedder", visual_frequency,
        prompt_input, cfg.hidden, visual_tokens);
    graph.action_condition_debug = ggml_dup(ctx, action_condition.modulation);
    ggml_tensor * action_condition_debug = graph.action_condition_debug;
    ggml_tensor * block0_action_debug = nullptr;

    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        const std::string block = "gwp.blocks." + std::to_string(layer);
        const std::string action_prefix = block + ".action_expert";
        const std::string visual_prefix = block + ".visual_expert";
        ggml_tensor * action_table = model.weight(action_prefix + ".scale_shift_table");
        ggml_tensor * visual_table = model.weight(visual_prefix + ".scale_shift_table");
        ggml_tensor * action_modulation = modulation_add(
            ctx, model, action_condition.modulation, action_table);
        ggml_tensor * visual_modulation = modulation_add(
            ctx, model, visual_condition.modulation, visual_table);
        ggml_tensor * a_shift = modulation_part(ctx, action_modulation, 0, cfg.expert_h, action_tokens);
        ggml_tensor * a_scale = modulation_part(ctx, action_modulation, 1, cfg.expert_h, action_tokens);
        ggml_tensor * a_gate = modulation_part(ctx, action_modulation, 2, cfg.expert_h, action_tokens);
        ggml_tensor * a_c_shift = modulation_part(ctx, action_modulation, 3, cfg.expert_h, action_tokens);
        ggml_tensor * a_c_scale = modulation_part(ctx, action_modulation, 4, cfg.expert_h, action_tokens);
        ggml_tensor * a_c_gate = modulation_part(ctx, action_modulation, 5, cfg.expert_h, action_tokens);
        ggml_tensor * v_shift = modulation_part(ctx, visual_modulation, 0, cfg.hidden, visual_tokens);
        ggml_tensor * v_scale = modulation_part(ctx, visual_modulation, 1, cfg.hidden, visual_tokens);
        ggml_tensor * v_gate = modulation_part(ctx, visual_modulation, 2, cfg.hidden, visual_tokens);
        ggml_tensor * v_c_shift = modulation_part(ctx, visual_modulation, 3, cfg.hidden, visual_tokens);
        ggml_tensor * v_c_scale = modulation_part(ctx, visual_modulation, 4, cfg.hidden, visual_tokens);
        ggml_tensor * v_c_gate = modulation_part(ctx, visual_modulation, 5, cfg.hidden, visual_tokens);

        ExpertQkv aqkv = expert_qkv(
            ctx, model, action_prefix, action_hidden, a_scale, a_shift,
            action_positions, visual_time, visual_height, visual_width, false);
        ExpertQkv vqkv = expert_qkv(
            ctx, model, visual_prefix, visual_hidden, v_scale, v_shift,
            action_positions, visual_time, visual_height, visual_width, true);
        ggml_tensor * q = ggml_concat(ctx, aqkv.q, vqkv.q, 2);
        ggml_tensor * k = ggml_concat(ctx, aqkv.k, vqkv.k, 2);
        ggml_tensor * v = ggml_concat(ctx, aqkv.v, vqkv.v, 2);
        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probabilities = ggml_soft_max_ext(
            ctx, scores, attention_mask,
            1.0f / std::sqrt(static_cast<float>(cfg.head_dim)), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, V, probabilities);
        attended = ggml_reshape_2d(
            ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)),
            cfg.hidden, total_tokens);
        ggml_tensor * action_attended = ggml_view_2d(
            ctx, attended, cfg.hidden, action_tokens, attended->nb[1], 0);
        ggml_tensor * visual_attended = ggml_view_2d(
            ctx, attended, cfg.hidden, visual_tokens, attended->nb[1],
            static_cast<size_t>(action_tokens) * attended->nb[1]);
        action_attended = linear_with_activation(
            ctx, model.weight(action_prefix + ".attn1.to_out.0.weight"),
            model.weight(action_prefix + ".attn1.to_out.0.bias"), action_attended,
            bf16_attention_value_output(model));
        visual_attended = linear_with_activation(
            ctx, model.weight(visual_prefix + ".attn1.to_out.0.weight"),
            model.weight(visual_prefix + ".attn1.to_out.0.bias"), visual_attended,
            bf16_attention_value_output(model));
        action_hidden = gated_residual(
            ctx, model, action_hidden, action_attended, a_gate);
        visual_hidden = gated_residual(
            ctx, model, visual_hidden, visual_attended, v_gate);
        action_hidden = expert_cross_ffn(
            ctx, model, action_prefix, action_hidden, action_condition.text,
            a_c_shift, a_c_scale, a_c_gate);
        visual_hidden = expert_cross_ffn(
            ctx, model, visual_prefix, visual_hidden, visual_condition.text,
            v_c_shift, v_c_scale, v_c_gate);
        if (layer == 0) block0_action_debug = ggml_dup(ctx, action_hidden);
    }

    ggml_tensor * output_temb = ggml_reshape_3d(
        ctx, action_condition.temb, cfg.expert_h, 1, action_tokens);
    ggml_tensor * output_modulation_shape = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, cfg.expert_h, 2, action_tokens);
    ggml_tensor * output_table = ggml_repeat(
        ctx, model.weight("gwp.action_scale_shift_table"), output_modulation_shape);
    ggml_tensor * output_modulation = modulation_add(
        ctx, model, output_table, output_temb);
    ggml_tensor * output_shift = ggml_view_2d(
        ctx, output_modulation, cfg.expert_h, action_tokens,
        output_modulation->nb[2], 0);
    ggml_tensor * output_scale = ggml_view_2d(
        ctx, output_modulation, cfg.expert_h, action_tokens,
        output_modulation->nb[2], output_modulation->nb[1]);
    ggml_tensor * normalized = modulated_norm(
        ctx, action_hidden, output_scale, output_shift, cfg.norm_eps);
    ggml_tensor * action_slice = ggml_view_2d(
        ctx, normalized, cfg.expert_h, cfg.action_chunk,
        normalized->nb[1], normalized->nb[1]);
    graph.block0_action_debug = block0_action_debug;
    graph.prediction = action_decoder(ctx, model, action_slice);
    if (!native_action_region(model)) {
        graph.prediction = ggml_cast(ctx, graph.prediction, GGML_TYPE_F32);
    }
    ggml_tensor * prediction = graph.prediction;
    graph.action_output = scheduler_step(
        ctx, model, action_input, prediction, graph.dt_input);
    ggml_set_output(graph.action_output);
    if (dump_intermediates && debug_dump_enabled()) {
        ggml_set_output(prediction);
        for (ggml_tensor * tensor : {action_tokens_debug, visual_tokens_debug,
                                     action_condition_debug, block0_action_debug}) {
            ggml_set_output(tensor);
        }
    }
    graph.cgraph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_cgraph * cgraph = graph.cgraph;
    ggml_build_forward_expand(cgraph, graph.action_output);
    if (dump_intermediates && debug_dump_enabled()) {
        ggml_build_forward_expand(cgraph, prediction);
        for (ggml_tensor * tensor : {action_tokens_debug, visual_tokens_debug,
                                     action_condition_debug, block0_action_debug}) {
            ggml_build_forward_expand(cgraph, tensor);
        }
    }
    ggml_graph_assign_uid(cgraph);
    graph.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, cgraph)) return false;
    audit_mixed_binary_nodes("complete-mot", cgraph);
    std::fprintf(stderr, "wam(gwp05): cached MoT graph (%.1f MiB)\n",
                ggml_gallocr_get_buffer_size(graph.alloc, 0) / (1024.0 * 1024.0));
    }

    ggml_tensor * state_input = graph.state_input;
    ggml_tensor * action_input = graph.action_input;
    ggml_tensor * reference_input = graph.reference_input;
    ggml_tensor * prompt_input = graph.prompt_input;
    ggml_tensor * action_frequency = graph.action_frequency;
    ggml_tensor * visual_frequency = graph.visual_frequency;
    ggml_tensor * action_positions = graph.action_positions;
    ggml_tensor * visual_time = graph.visual_time;
    ggml_tensor * visual_height = graph.visual_height;
    ggml_tensor * visual_width = graph.visual_width;
    ggml_tensor * attention_mask = graph.attention_mask;
    ggml_tensor * dt_input = graph.dt_input;
    ggml_tensor * prediction = graph.prediction;
    ggml_tensor * action_output = graph.action_output;
    ggml_tensor * action_tokens_debug = graph.action_tokens_debug;
    ggml_tensor * visual_tokens_debug = graph.visual_tokens_debug;
    ggml_tensor * action_condition_debug = graph.action_condition_debug;
    ggml_tensor * block0_action_debug = graph.block0_action_debug;
    ggml_cgraph * cgraph = graph.cgraph;

    std::vector<float> action_frequencies(static_cast<size_t>(256) * action_tokens);
    std::vector<float> visual_frequencies(static_cast<size_t>(256) * visual_tokens);
    const auto action_timestep_values =
        gwp05_semantics::action_timesteps(cfg.action_chunk, timestep, true);
    for (int64_t token = 0; token < action_tokens; ++token) {
        const std::vector<float> embedding =
            timestep_embedding(action_timestep_values[static_cast<size_t>(token)], 256);
        std::copy(embedding.begin(), embedding.end(), action_frequencies.begin() + token * 256);
    }
    const std::vector<float> zero_embedding = timestep_embedding(0.0f, 256);
    for (int64_t token = 0; token < visual_tokens; ++token) {
        std::copy(zero_embedding.begin(), zero_embedding.end(),
                  visual_frequencies.begin() + token * 256);
    }
    const std::vector<int32_t> action_position_values =
        gwp05_semantics::action_positions(cfg.action_chunk, true);
    const auto visual_positions = gwp05_semantics::visual_positions(grid_height, grid_width);
    std::vector<float> mask(static_cast<size_t>(total_tokens) * total_tokens * cfg.n_q_heads);
    for (int64_t head = 0; head < cfg.n_q_heads; ++head) {
        for (int64_t query = 0; query < total_tokens; ++query) {
            for (int64_t key = 0; key < total_tokens; ++key) {
                const bool query_is_action = query > 0 && query < action_tokens;
                const bool key_is_action = key > 0 && key < action_tokens;
                const bool allowed = query_is_action || !key_is_action;
                const size_t index =
                    (static_cast<size_t>(head) * total_tokens + query) * total_tokens + key;
                mask[index] = allowed ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
    }

    ggml_backend_tensor_set(state_input, state.data(), 0, state.size() * sizeof(float));
    ggml_backend_tensor_set(reference_input, reference.data(), 0, reference.size() * sizeof(float));
    ggml_backend_tensor_set(prompt_input, prompt.data(), 0, prompt.size() * sizeof(float));
    if (upload_action || built_graph) {
        set_f32_tensor(action_input, action);
    }
    ggml_backend_tensor_set(dt_input, &dt, 0, sizeof(dt));
    ggml_backend_tensor_set(action_frequency, action_frequencies.data(), 0,
                            action_frequencies.size() * sizeof(float));
    ggml_backend_tensor_set(visual_frequency, visual_frequencies.data(), 0,
                            visual_frequencies.size() * sizeof(float));
    ggml_backend_tensor_set(action_positions, action_position_values.data(), 0,
                            action_position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set(visual_time, visual_positions.time.data(), 0,
                            visual_positions.time.size() * sizeof(int32_t));
    ggml_backend_tensor_set(visual_height, visual_positions.height.data(), 0,
                            visual_positions.height.size() * sizeof(int32_t));
    ggml_backend_tensor_set(visual_width, visual_positions.width.data(), 0,
                            visual_positions.width.size() * sizeof(int32_t));
    ggml_backend_tensor_set(attention_mask, mask.data(), 0, mask.size() * sizeof(float));
    if (ggml_backend_graph_compute(model.backend, cgraph) != GGML_STATUS_SUCCESS) return false;
    if (dump_intermediates) {
        debug_dump_tensor("action_tokens", action_tokens_debug);
        debug_dump_tensor("visual_tokens", visual_tokens_debug);
        debug_dump_tensor("action_condition", action_condition_debug);
        debug_dump_tensor("block0_action", block0_action_debug);
    }
    if (prediction_output) {
        get_f32_tensor(prediction, *prediction_output);
    }
    ggml_backend_tensor_copy(action_output, action_input);
    ggml_backend_synchronize(model.backend);
    if (action_output_host) {
        get_f32_tensor(action_output, *action_output_host);
    }
    if (std::getenv("WAM_GWP05_DISABLE_MOT_GRAPH_CACHE")) model.mot_graph.reset();
    return true;
}
