#include "models/gwp05/state.h"

namespace wam::internal::gwp05 {

void ExecutionState::reset() {
    mot_graph.reset();
    unrolled_action_graph.reset();
    cached_action_graph.reset();
    prefix_graph.reset();
    prefix_storage.reset();
    prompt_projection_graph.reset();
    prompt_cache.clear();
    projected_prompt_signature.clear();
    prompt_cache_hits = 0;
    prompt_cache_misses = 0;
    projected_prompt_hits = 0;
    projected_prompt_misses = 0;
    stats = {};
}

ExecutionState::~ExecutionState() {
    mot_graph.reset();
    unrolled_action_graph.reset();
    cached_action_graph.reset();
    prompt_projection_graph.reset();
    prefix_graph.reset();
    prefix_storage.reset();
    vae_graph.reset();
    backend = nullptr;
    weights.clear();
}

} // namespace wam::internal::gwp05
