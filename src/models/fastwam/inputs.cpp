#include "models/fastwam/inputs.h"

#include "policy/language_ops.h"
#include "policy/observation_ops.h"

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
    prepared.observation = policy::prepare_observation_reference(
        inputs, policy_spec, session_rng);

    policy::PreparedEmbedding embedding = policy::prepare_embedding_input(
        *input, policy_spec.language, artifact.geometry.text_dim);
    prepared.embedding = std::move(embedding.embedding);
    prepared.embedding_attention_mask = std::move(embedding.attention_mask);
    return prepared;
}

} // namespace wam::internal::fastwam
