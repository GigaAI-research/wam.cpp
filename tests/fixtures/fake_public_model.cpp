#include "fixtures/fake_public_model.h"

#include "model_internal.h"
#include "wam/error.h"
#include "wam/session.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace wam::test {
namespace {

class FakeSession final : public internal::SessionImpl {
public:
    FakeSession(std::shared_ptr<FakeModelCounters> counters,
                std::uint64_t reset_mode)
        : counters_(std::move(counters)), reset_mode_(reset_mode) {}

    ~FakeSession() override {
        ++counters_->sessions_destroyed;
    }

    Prediction predict(const Observation &) override {
        ++counters_->predictions;
        Prediction prediction;
        prediction.action.data = {4, 2};
        prediction.action.dtype = DType::u8;
        prediction.action.shape = {1, 2};
        return prediction;
    }

    void reset() override {
        if (reset_mode_ == 1) {
            throw Error(ErrorCode::failed_precondition, "fake reset failure",
                        {{"session", "test"}});
        }
        if (reset_mode_ == 2) throw std::runtime_error("standard failure");
        if (reset_mode_ == 3) throw 3;
    }

private:
    std::shared_ptr<FakeModelCounters> counters_;
    std::uint64_t reset_mode_;
};

class FakeModel final : public internal::ModelImpl {
public:
    FakeModel(std::shared_ptr<FakeModelCounters> counters,
              bool return_null_session)
        : ModelImpl(make_info(), make_policy_spec()),
          counters_(std::move(counters)),
          return_null_session_(return_null_session) {}

    ~FakeModel() override {
        ++counters_->models_destroyed;
    }

    std::unique_ptr<internal::SessionImpl> create_session(
        const SessionConfig & config) override {
        if (return_null_session_) return nullptr;
        ++counters_->sessions_created;
        return std::make_unique<FakeSession>(counters_, config.random_seed);
    }

private:
    static ModelInfo make_info() {
        ModelInfo info;
        info.architecture = "test-only";
        info.capabilities.action = true;
        info.capabilities.concurrent_sessions = true;
        return info;
    }

    static PolicySpec make_policy_spec() {
        PolicySpec spec;
        spec.identity.profile = "fake-v1";
        return spec;
    }

    std::shared_ptr<FakeModelCounters> counters_;
    bool return_null_session_;
};

} // namespace

Model make_fake_model(const std::shared_ptr<FakeModelCounters> & counters,
                      bool return_null_session) {
    return internal::adopt_model(
        std::make_unique<FakeModel>(counters, return_null_session));
}

} // namespace wam::test
