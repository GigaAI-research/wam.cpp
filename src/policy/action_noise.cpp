#include "policy/action_noise.h"

#include "policy/tensor_input.h"
#include "wam/error.h"

#include <limits>

namespace wam::internal::policy {
namespace {

bool omitted_tensor(const TensorView & tensor) {
    return tensor.data == nullptr && tensor.byte_size == 0 &&
        tensor.dtype == DType::unknown && tensor.shape.empty() &&
        tensor.layout.empty() &&
        tensor.byte_order == ByteOrder::not_applicable;
}

} // namespace

std::vector<float> prepare_action_noise(const TensorView & explicit_noise,
                                        const ActionSpec & spec,
                                        std::mt19937 & session_rng) {
    if (!omitted_tensor(explicit_noise)) {
        if (!explicit_noise.layout.empty() &&
            explicit_noise.layout != "T,A") {
            throw Error(ErrorCode::invalid_argument,
                        "action noise tensor layout is invalid",
                        {{"action_noise.layout", "expected T,A or empty"}});
        }
        return copy_f32_tensor(
            explicit_noise,
            {{static_cast<std::int64_t>(spec.horizon),
              static_cast<std::int64_t>(spec.model_dim)}},
            "action_noise");
    }
    if (spec.horizon == 0 || spec.model_dim == 0 ||
        spec.model_dim > std::numeric_limits<std::size_t>::max() /
                             spec.horizon) {
        throw Error(ErrorCode::invalid_argument,
                    "action noise shape is invalid");
    }
    std::vector<float> noise(spec.horizon * spec.model_dim);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float & value : noise) value = normal(session_rng);
    return noise;
}

} // namespace wam::internal::policy
