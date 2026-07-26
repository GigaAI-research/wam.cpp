#include "models/fastwam/inputs.h"

#include "policy/action_ops.h"
#include "policy/image_ops.h"
#include "policy/language_ops.h"
#include "policy/state_ops.h"

#include <string>

namespace wam::internal::fastwam {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

} // namespace

PreparedInputs prepare_inputs(const Inputs & inputs,
                              const ArtifactContract & artifact,
                              const policy::PolicySpecDraft & policy_spec,
                              LanguageRuntimeMode language_mode,
                              std::mt19937 & session_rng) {
    if (!inputs.history.empty()) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM Gate B does not expose history tensors",
                    {{"history", "must be empty"}});
    }
    if (language_mode != LanguageRuntimeMode::external_embedding) {
        throw Error(ErrorCode::failed_precondition,
                    "FastWAM requires external embedding language mode",
                    {{"language_mode", "expected external_embedding"}});
    }
    const auto * input = std::get_if<EmbeddingInput>(&inputs.language);
    if (input == nullptr) {
        invalid("FastWAM embedding language input is required", "language",
                "expected external_embedding");
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

    policy::PreparedEmbedding embedding = policy::prepare_embedding_input(
        *input, policy_spec.language, artifact.geometry.text_dim);
    prepared.embedding = std::move(embedding.embedding);
    prepared.embedding_attention_mask = std::move(embedding.attention_mask);
    prepared.action_noise = policy::prepare_action_noise(
        inputs.action_noise, policy_spec.action, session_rng);
    return prepared;
}

} // namespace wam::internal::fastwam
