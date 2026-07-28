#pragma once

#include "wam/observation.h"
#include "wam/prediction.h"

#include <memory>

namespace wam {

class Model;

namespace internal {
class ModelImpl;
class SessionImpl;
} // namespace internal

class Session final {
public:
    ~Session();
    Session(Session &&) noexcept;
    Session & operator=(Session &&) noexcept;
    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;

    Prediction predict(const Observation & observation);
    void reset();

private:
    Session(std::shared_ptr<internal::ModelImpl> model,
            std::unique_ptr<internal::SessionImpl> impl);

    std::shared_ptr<internal::ModelImpl> model_;
    std::unique_ptr<internal::SessionImpl> impl_;
    friend class Model;
};

} // namespace wam
