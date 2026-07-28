#include "wam/pipeline.h"

#include <utility>

namespace wam {

Pipeline::Pipeline(Model model, Session session)
    : model_(std::move(model)), session_(std::move(session)) {}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline &&) noexcept = default;
Pipeline & Pipeline::operator=(Pipeline &&) noexcept = default;

Pipeline Pipeline::load(const std::string & artifact_path,
                        const RuntimeConfig & runtime_config,
                        const SessionConfig & session_config) {
    Model model = Model::load(artifact_path, runtime_config);
    Session session = model.create_session(session_config);
    return Pipeline(std::move(model), std::move(session));
}

Prediction Pipeline::predict(const Observation & observation) {
    return session_.predict(observation);
}

void Pipeline::reset() {
    session_.reset();
}

const ModelInfo & Pipeline::model_info() const {
    return model_.info();
}

} // namespace wam
