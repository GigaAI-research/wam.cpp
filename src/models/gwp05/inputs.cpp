#include "models/gwp05/inputs.h"

#include "models/common/input_validation.h"
#include "models/gwp05/semantics.h"
#include "policy/image_ops.h"
#include "policy/state_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace wam::internal::gwp05 {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

[[noreturn]] void unsupported(const std::string & message,
                              const std::string & field,
                              const std::string & reason) {
    throw Error(ErrorCode::unsupported, message, {{field, reason}});
}

bool supports_tokens(policy::LanguageInputMode mode) {
    return mode == policy::LanguageInputMode::tokens ||
        mode == policy::LanguageInputMode::tokens_or_embedding;
}

bool supports_embedding(policy::LanguageInputMode mode) {
    return mode == policy::LanguageInputMode::embedding ||
        mode == policy::LanguageInputMode::tokens_or_embedding;
}

semantics::PromptPlan validate_tokens(
    const TokenInput & input, const ArtifactContract & artifact,
    const policy::LanguageSpec & language,
    std::vector<std::int32_t> & token_ids,
    std::vector<std::int32_t> & attention_mask) {
    if (input.token_ids.size == 0 || input.token_ids.data == nullptr) {
        invalid("token input is empty", "language.token_ids",
                "nonempty token IDs are required");
    }
    if (input.token_ids.size > language.max_tokens) {
        invalid("token input exceeds PolicySpec capacity",
                "language.token_ids",
                std::to_string(input.token_ids.size));
    }
    token_ids.assign(input.token_ids.data,
                     input.token_ids.data + input.token_ids.size);
    if (input.attention_mask.empty()) {
        if (language.attention_mask_required) {
            invalid("token attention mask is required",
                    "language.attention_mask", "missing");
        }
        attention_mask.assign(token_ids.size(), 1);
    } else {
        if (input.attention_mask.data == nullptr ||
            input.attention_mask.size != token_ids.size()) {
            invalid("token IDs and attention mask lengths differ",
                    "language.attention_mask", "shape mismatch");
        }
        attention_mask.assign(
            input.attention_mask.data,
            input.attention_mask.data + input.attention_mask.size);
    }
    return semantics::prepare_prompt(
        token_ids, attention_mask, artifact.geometry.t5_vocab_size,
        language.padding_side);
}

void prepare_token_language(const TokenInput & input,
                            const ArtifactContract & artifact,
                            const policy::LanguageSpec & language,
                            PreparedInputs & prepared) {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> mask;
    const semantics::PromptPlan plan = validate_tokens(
        input, artifact, language, token_ids, mask);
    prepared.token_ids = plan.active_token_ids;
    prepared.attention_mask.assign(prepared.token_ids.size(), 1);
}

void validate_bf16_finite(const TensorView & tensor,
                          const std::string & field) {
    const std::size_t elements = checked_numel(tensor.shape, field);
    const auto * bytes = static_cast<const std::uint8_t *>(tensor.data);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t value = 0;
        std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
        if ((value & 0x7F80U) == 0x7F80U) {
            invalid("BF16 tensor contains NaN or Inf", field,
                    std::to_string(index));
        }
    }
}

void prepare_embedding_language(
    const EmbeddingInput & input, const ArtifactContract & artifact,
    const policy::LanguageSpec & language, PreparedInputs & prepared) {
    const TensorView & embedding = input.embedding;
    if (embedding.data == nullptr || embedding.shape.size() != 2 ||
        embedding.shape[0] <= 0 ||
        embedding.shape[1] !=
            static_cast<std::int64_t>(artifact.geometry.t5_hidden) ||
        embedding.shape[0] >
            static_cast<std::int64_t>(language.max_tokens) ||
        embedding.layout != "T,D" ||
        embedding.byte_order != ByteOrder::little ||
        (embedding.dtype != DType::f32 &&
         embedding.dtype != DType::bf16)) {
        invalid("prompt embedding contract is invalid", "language.embedding",
                "expected little-endian F32/BF16 [T,D] with layout T,D");
    }
    const std::size_t elements =
        checked_numel(embedding.shape, "language.embedding");
    const std::size_t width = dtype_size(embedding.dtype);
    if (width == 0 ||
        elements > std::numeric_limits<std::size_t>::max() / width ||
        embedding.byte_size != elements * width) {
        invalid("prompt embedding byte size is invalid", "language.embedding",
                "payload size mismatch");
    }
    if (embedding.dtype == DType::f32) {
        const auto * bytes =
            static_cast<const std::uint8_t *>(embedding.data);
        for (std::size_t index = 0; index < elements; ++index) {
            float value = 0.0F;
            std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
            if (!std::isfinite(value)) {
                invalid("prompt embedding contains NaN or Inf",
                        "language.embedding", std::to_string(index));
            }
        }
    } else {
        validate_bf16_finite(embedding, "language.embedding");
    }

    const std::size_t tokens =
        static_cast<std::size_t>(embedding.shape[0]);
    if (input.attention_mask.empty()) {
        if (language.attention_mask_required) {
            invalid("embedding attention mask is required",
                    "language.attention_mask", "missing");
        }
        prepared.embedding_attention_mask.assign(tokens, 1);
    } else {
        prepared.embedding_attention_mask = copy_i32_tensor(
            input.attention_mask,
            {static_cast<std::int64_t>(tokens)},
            "language.attention_mask");
    }
    const std::vector<std::int32_t> dummy_tokens(tokens, 0);
    (void) semantics::prepare_prompt(
        dummy_tokens, prepared.embedding_attention_mask, 1,
        language.padding_side);

    prepared.embedding.dtype = embedding.dtype;
    prepared.embedding.shape = embedding.shape;
    prepared.embedding.layout = embedding.layout;
    prepared.embedding.byte_order = embedding.byte_order;
    prepared.embedding.data.resize(embedding.byte_size);
    std::memcpy(prepared.embedding.data.data(), embedding.data,
                embedding.byte_size);
}

TokenInput fixed_prompt_view(const FixedPrompt & prompt) {
    return {ArrayView<std::int32_t>(prompt.token_ids),
            ArrayView<std::int32_t>(prompt.attention_mask)};
}

void prepare_language(const Inputs & inputs,
                      const ArtifactContract & artifact,
                      const policy::PolicySpecDraft & policy_spec,
                      LanguageRuntimeMode language_mode,
                      const std::optional<FixedPrompt> & fixed_prompt,
                      PreparedInputs & prepared) {
    if (language_mode == LanguageRuntimeMode::automatic) {
        throw Error(ErrorCode::failed_precondition,
                    "GWP language mode must be resolved at model load",
                    {{"language_mode", "automatic"}});
    }
    const policy::LanguageSpec & language = policy_spec.language;
    if (fixed_prompt.has_value()) {
        if (language_mode != LanguageRuntimeMode::tokens) {
            throw Error(ErrorCode::failed_precondition,
                        "fixed prompt requires token language mode");
        }
        if (const auto * supplied =
                std::get_if<TokenInput>(&inputs.language)) {
            const FixedPrompt & fixed = *fixed_prompt;
            const bool matches =
                supplied->token_ids.size == fixed.token_ids.size() &&
                supplied->attention_mask.size ==
                    fixed.attention_mask.size() &&
                supplied->token_ids.data != nullptr &&
                supplied->attention_mask.data != nullptr &&
                std::equal(fixed.token_ids.begin(), fixed.token_ids.end(),
                           supplied->token_ids.data) &&
                std::equal(fixed.attention_mask.begin(),
                           fixed.attention_mask.end(),
                           supplied->attention_mask.data);
            if (!matches) {
                throw Error(ErrorCode::failed_precondition,
                            "request token input conflicts with fixed prompt");
            }
        } else if (!std::holds_alternative<std::monostate>(inputs.language)) {
            throw Error(ErrorCode::failed_precondition,
                        "fixed prompt cannot be combined with embedding input");
        }
        prepare_token_language(fixed_prompt_view(*fixed_prompt), artifact,
                               language, prepared);
        return;
    }

    if (language_mode == LanguageRuntimeMode::tokens) {
        if (!supports_tokens(language.input_mode)) {
            throw Error(ErrorCode::failed_precondition,
                        "PolicySpec does not allow token input");
        }
        const auto * token_input = std::get_if<TokenInput>(&inputs.language);
        if (token_input == nullptr) {
            invalid("GWP token language input is required", "language",
                    "expected tokens");
        }
        prepare_token_language(*token_input, artifact, language, prepared);
        return;
    }

    if (language_mode == LanguageRuntimeMode::external_embedding) {
        if (!supports_embedding(language.input_mode)) {
            throw Error(ErrorCode::failed_precondition,
                        "PolicySpec does not allow embedding input");
        }
        const auto * embedding =
            std::get_if<EmbeddingInput>(&inputs.language);
        if (embedding == nullptr) {
            invalid("GWP embedding language input is required", "language",
                    "expected external_embedding");
        }
        prepare_embedding_language(*embedding, artifact, language, prepared);
        return;
    }
    throw Error(ErrorCode::failed_precondition,
                "GWP language mode is invalid");
}

} // namespace

PreparedInputs prepare_inputs(
    const Inputs & inputs, const ArtifactContract & artifact,
    const policy::PolicySpecDraft & policy_spec,
    LanguageRuntimeMode language_mode,
    std::mt19937 & session_rng,
    const std::optional<FixedPrompt> & fixed_prompt) {
    if (!inputs.history.empty()) {
        unsupported("GWP Slice 3 does not expose history tensors",
                    "history", "must be empty");
    }

    PreparedInputs prepared;
    const std::vector<std::size_t> order = policy::resolve_image_order(
        inputs.images, policy_spec.images);
    std::vector<policy::CpuImage> transformed;
    transformed.reserve(order.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        transformed.push_back(policy::transform_image_reference(
            inputs.images[order[index]], policy_spec.images.views[index],
            policy_spec.images));
    }
    prepared.composite_image = policy::compose_canvas_reference(
        transformed, policy_spec.images);

    policy::validate_state_input(inputs.state, policy_spec.state);
    prepared.raw_state = policy::read_state_f32(inputs.state);
    prepared.model_state = policy::pad_state(
        prepared.raw_state, policy_spec.state);
    policy::normalize_state_reference(
        prepared.model_state, policy_spec.state, policy_spec.state.stats);

    prepared.language_mode = language_mode;
    prepare_language(inputs, artifact, policy_spec, language_mode,
                     fixed_prompt, prepared);

    prepared.action_noise = policy::prepare_action_noise(
        inputs.action_noise, policy_spec.action, session_rng);
    return prepared;
}

} // namespace wam::internal::gwp05
