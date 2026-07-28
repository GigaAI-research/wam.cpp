#include "policy/observation_processor.h"

#include "policy/action_noise.h"
#include "policy/state_ops.h"

namespace wam::internal::policy {

PreparedObservation prepare_observation_reference(
    const Observation & inputs, const PolicySpec & policy_spec,
    std::mt19937 & session_rng) {
    PreparedObservation prepared;
    const std::vector<std::size_t> order = resolve_image_order(
        inputs.images, policy_spec.images);
    std::vector<CpuImage> transformed;
    transformed.reserve(order.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        transformed.push_back(transform_image_reference(
            inputs.images[order[index]], policy_spec.images.views[index],
            policy_spec.images));
    }
    prepared.composite_image = compose_canvas_reference(
        transformed, policy_spec.images);

    validate_state_input(inputs.state, policy_spec.state);
    prepared.raw_state = read_state_f32(inputs.state);
    prepared.model_state = pad_state(prepared.raw_state, policy_spec.state);
    normalize_state_reference(
        prepared.model_state, policy_spec.state, policy_spec.state.stats);
    prepared.action_noise = prepare_action_noise(
        inputs.action_noise, policy_spec.action, session_rng);
    return prepared;
}

} // namespace wam::internal::policy
