#pragma once

#include "policy/policy_spec.h"
#include "wam/observation.h"
#include "wam/prediction.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wam::internal::policy {

struct PreparedEmbedding {
    Tensor embedding;
    std::vector<std::int32_t> attention_mask;
};

PreparedEmbedding prepare_embedding_input(const EmbeddingInput & input,
                                          const LanguageSpec & spec,
                                          std::size_t expected_width);

} // namespace wam::internal::policy
