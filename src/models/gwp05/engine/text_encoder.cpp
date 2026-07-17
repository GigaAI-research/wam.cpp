std::vector<float> timestep_embedding(float timestep, int64_t dim) {
    const int64_t half = dim / 2;
    std::vector<float> output(dim, 0.0f);
    for (int64_t i = 0; i < half; ++i) {
        const float frequency = std::exp(-std::log(10000.0f) * static_cast<float>(i) /
                                         static_cast<float>(half));
        const float value = timestep * frequency;
        output[i] = std::cos(value);       // flip_sin_to_cos=True
        output[i + half] = std::sin(value);
    }
    return output;
}

int relative_position_bucket(int relative, int buckets, int max_distance) {
    const int half = buckets / 2;
    int bucket = relative > 0 ? half : 0;
    const int distance = std::abs(relative);
    const int max_exact = half / 2;
    if (distance < max_exact) return bucket + distance;
    const float ratio = std::log(static_cast<float>(distance) / max_exact) /
                        std::log(static_cast<float>(max_distance) / max_exact);
    const int large = std::min(half - 1, max_exact + static_cast<int>(ratio * (half - max_exact)));
    return bucket + large;
}

ggml_tensor * build_attention(
        ggml_context * ctx,
        ggml_tensor * query_input,
        ggml_tensor * key_value_input,
        ggml_tensor * query_weight,
        ggml_tensor * query_bias,
        ggml_tensor * key_weight,
        ggml_tensor * key_bias,
        ggml_tensor * value_weight,
        ggml_tensor * value_bias,
        ggml_tensor * output_weight,
        ggml_tensor * output_bias,
        ggml_tensor * query_norm,
        ggml_tensor * key_norm,
        int64_t heads,
        int64_t head_dim,
        ggml_tensor * mask,
        float scale,
        float eps,
        bool use_bf16_qkv = false,
        bool use_bf16_value_output = false) {
    const int64_t query_tokens = query_input->ne[1];
    const int64_t key_tokens = key_value_input->ne[1];
    ggml_tensor * q = linear_with_activation(
        ctx, query_weight, query_bias, query_input, use_bf16_qkv);
    ggml_tensor * k = linear_with_activation(
        ctx, key_weight, key_bias, key_value_input, use_bf16_qkv);
    ggml_tensor * v = linear_with_activation(
        ctx, value_weight, value_bias, key_value_input, use_bf16_qkv);
    if (query_norm) q = rms_norm(ctx, q, query_norm, eps);
    if (key_norm) k = rms_norm(ctx, k, key_norm, eps);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, query_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, key_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, key_tokens);
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    if (use_bf16_value_output) V = ggml_cast(ctx, V, GGML_TYPE_BF16);
    ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * probabilities = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * attended = ggml_mul_mat(ctx, V, probabilities);
    attended = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)),
        heads * head_dim, query_tokens);
    return linear_with_activation(
        ctx, output_weight, output_bias, attended, use_bf16_value_output);
}

bool validate_prompt_mask(const EngineInputsView & in, int & valid_tokens) {
    if (!in.attention_mask && in.attention_mask_n != 0) {
        std::fprintf(stderr, "wam(gwp05): T5 mask length is nonzero without a mask\n");
        return false;
    }
    if (in.attention_mask && in.attention_mask_n != in.n_lang) {
        std::fprintf(stderr, "wam(gwp05): T5 mask length %d does not match token count %d\n",
                     in.attention_mask_n, in.n_lang);
        return false;
    }
    valid_tokens = in.n_lang;
    if (in.attention_mask) {
        valid_tokens = 0;
        bool saw_padding = false;
        for (int i = 0; i < in.n_lang; ++i) {
            if (in.attention_mask[i] != 0 && in.attention_mask[i] != 1) {
                std::fprintf(stderr, "wam(gwp05): T5 mask values must be zero or one\n");
                return false;
            }
            if (in.attention_mask[i] == 1) {
                if (saw_padding) {
                    std::fprintf(stderr, "wam(gwp05): T5 mask must be a contiguous valid prefix\n");
                    return false;
                }
                ++valid_tokens;
            } else {
                saw_padding = true;
            }
        }
    }
    if (valid_tokens == 0) {
        std::fprintf(stderr, "wam(gwp05): T5 mask contains no valid tokens\n");
        return false;
    }
    return true;
}

std::vector<float> run_t5(Gwp05ModelArch & model, const EngineInputsView & in) {
    if (!component_is_loaded(model, WeightComponent::t5)) {
        std::fprintf(stderr, "wam(gwp05): T5 requested while its weights are not resident\n");
        return {};
    }
    const Config & cfg = model.cfg;
    if (!in.lang_tokens || in.n_lang < 1 || in.n_lang > cfg.n_lang) {
        std::fprintf(stderr, "wam(gwp05): lang token count %d outside [1,%lld]\n",
                     in.n_lang, static_cast<long long>(cfg.n_lang));
        return {};
    }
    int valid_tokens = 0;
    if (!validate_prompt_mask(in, valid_tokens)) return {};
    for (int i = 0; i < valid_tokens; ++i) {
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= cfg.t5_vocab_size) {
            std::fprintf(stderr, "wam(gwp05): token %d is outside T5 vocabulary\n", in.lang_tokens[i]);
            return {};
        }
    }
    GraphContext graph;
    ggml_init_params params{};
    params.mem_size = 128u * 1024u * 1024u;
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return {};
    ggml_context * ctx = graph.ctx;
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, valid_tokens);
    ggml_set_input(ids);

    ggml_tensor * hidden = ggml_get_rows(ctx, model.weight("t5.shared.weight"), ids);
    std::vector<int32_t> bucket_ids(static_cast<size_t>(valid_tokens) * valid_tokens);
    const int buckets = 32;
    const int max_distance = 128;
    for (int query = 0; query < valid_tokens; ++query) {
        for (int key = 0; key < valid_tokens; ++key) {
            bucket_ids[static_cast<size_t>(query) * valid_tokens + key] =
                relative_position_bucket(key - query, buckets, max_distance);
        }
    }
    ggml_tensor * bucket_tensor = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, static_cast<int64_t>(bucket_ids.size()));
    ggml_set_input(bucket_tensor);
    for (int64_t layer = 0; layer < cfg.t5_layers; ++layer) {
        const std::string prefix = "t5.encoder.block." + std::to_string(layer) + ".layer";
        ggml_tensor * position_bias = ggml_get_rows(
            ctx,
            model.weight(prefix + ".0.SelfAttention.relative_attention_bias.weight"),
            bucket_tensor);
        position_bias = ggml_reshape_3d(
            ctx, position_bias, cfg.t5_heads, valid_tokens, valid_tokens);
        position_bias = ggml_cont(
            ctx, ggml_permute(ctx, position_bias, 2, 0, 1, 3));
        ggml_tensor * normalized = rms_norm(
            ctx, hidden, model.weight(prefix + ".0.layer_norm.weight"), 1e-6f);
        const std::string attention = prefix + ".0.SelfAttention";
        ggml_tensor * attention_output = build_attention(
            ctx, normalized, normalized,
            model.weight(attention + ".q.weight"), nullptr,
            model.weight(attention + ".k.weight"), nullptr,
            model.weight(attention + ".v.weight"), nullptr,
            model.weight(attention + ".o.weight"), nullptr,
            nullptr, nullptr, cfg.t5_heads, cfg.t5_head_dim,
            position_bias, 1.0f, 1e-6f);
        hidden = ggml_add(ctx, hidden, attention_output);

        normalized = rms_norm(
            ctx, hidden, model.weight(prefix + ".1.layer_norm.weight"), 1e-6f);
        const std::string ffn = prefix + ".1.DenseReluDense";
        ggml_tensor * gated = ggml_gelu(
            ctx, ggml_mul_mat(ctx, model.weight(ffn + ".wi_0.weight"), normalized));
        ggml_tensor * linear_part = ggml_mul_mat(
            ctx, model.weight(ffn + ".wi_1.weight"), normalized);
        ggml_tensor * ffn_output = ggml_mul_mat(
            ctx, model.weight(ffn + ".wo.weight"), ggml_mul(ctx, gated, linear_part));
        hidden = ggml_add(ctx, hidden, ffn_output);
    }
    hidden = rms_norm(ctx, hidden, model.weight("t5.encoder.final_layer_norm.weight"), 1e-6f);
    hidden = ggml_cast(ctx, hidden, GGML_TYPE_F32);
    ggml_set_output(hidden);
    ggml_cgraph * cgraph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(cgraph, hidden);
    graph.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, cgraph)) return {};
    ggml_backend_tensor_set(ids, in.lang_tokens, 0, valid_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(bucket_tensor, bucket_ids.data(), 0,
                            bucket_ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(model.backend, cgraph) != GGML_STATUS_SUCCESS) return {};

    std::vector<float> output(static_cast<size_t>(cfg.n_lang) * cfg.t5_hidden, 0.0f);
    ggml_backend_tensor_get(hidden, output.data(), 0,
                            static_cast<size_t>(valid_tokens) * cfg.t5_hidden * sizeof(float));
    return output;
}
