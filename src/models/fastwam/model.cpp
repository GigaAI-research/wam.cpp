#include "models/fastwam/model.h"

#include "arch.h"
#include "model_registry.h"

namespace wam::internal::fastwam {

std::unique_ptr<ModelImpl> create_model(
    const ModelOptions & options,
    ModelInfo info,
    std::optional<policy::PolicySpecDraft> policy_spec,
    std::shared_ptr<GgufReader> reader) {
    (void) options;
    (void) info;
    (void) policy_spec;
    (void) reader;
    throw Error(ErrorCode::unsupported,
                "FastWAM artifact execution is not implemented until Slice 8",
                {{"general.architecture", "fastwam"}});
}

} // namespace wam::internal::fastwam

namespace wam::internal {

void register_fastwam(ModelRegistry & registry) {
    registry.add(Arch::fastwam, fastwam::create_model);
}

} // namespace wam::internal
