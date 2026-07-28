#pragma once

#include "wam/model.h"
#include "wam/session.h"

namespace wam {

class Pipeline final {
public:
    static Pipeline load(const std::string & artifact_path,
                         const RuntimeConfig & runtime_config = {},
                         const SessionConfig & session_config = {});

    ~Pipeline();
    Pipeline(Pipeline &&) noexcept;
    Pipeline & operator=(Pipeline &&) noexcept;
    Pipeline(const Pipeline &) = delete;
    Pipeline & operator=(const Pipeline &) = delete;

    Prediction predict(const Observation & observation);
    void reset();
    const ModelInfo & model_info() const;

private:
    Pipeline(Model model, Session session);

    Model model_;
    Session session_;
};

} // namespace wam
