#include "support/test_utils.h"

#include "model_internal.h"
#include "wam/wam.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

struct Counters {
    int models_destroyed = 0;
    int sessions_created = 0;
    int sessions_destroyed = 0;
    int predictions = 0;
};

class FakeSession final : public wam::internal::SessionImpl {
public:
    FakeSession(std::shared_ptr<Counters> counters, std::uint64_t reset_mode)
        : counters_(std::move(counters)), reset_mode_(reset_mode) {}

    ~FakeSession() override {
        ++counters_->sessions_destroyed;
    }

    wam::Prediction predict(const wam::Inputs &) override {
        ++counters_->predictions;
        wam::Prediction prediction;
        prediction.action.data = {4, 2};
        prediction.action.dtype = wam::DType::u8;
        prediction.action.shape = {2};
        return prediction;
    }

    wam::Status reset() override {
        if (reset_mode_ == 1) {
            throw wam::Error(wam::ErrorCode::failed_precondition,
                             "wam reset failure", {{"session", "test"}});
        }
        if (reset_mode_ == 2) {
            throw std::runtime_error("standard reset failure");
        }
        if (reset_mode_ == 3) {
            throw 3;
        }
        return wam::Status::success();
    }

private:
    std::shared_ptr<Counters> counters_;
    std::uint64_t reset_mode_;
};

class FakeModel final : public wam::internal::ModelImpl {
public:
    explicit FakeModel(std::shared_ptr<Counters> counters,
                       bool return_null_session = false)
        : ModelImpl(make_info(), {}),
          counters_(std::move(counters)),
          return_null_session_(return_null_session) {}

    ~FakeModel() override {
        ++counters_->models_destroyed;
    }

    std::unique_ptr<wam::internal::SessionImpl> create_session(
        const wam::SessionOptions & options) override {
        if (return_null_session_) {
            return nullptr;
        }
        ++counters_->sessions_created;
        return std::make_unique<FakeSession>(counters_, options.random_seed);
    }

private:
    static wam::ModelInfo make_info() {
        wam::ModelInfo info;
        info.architecture = "test-only";
        info.capabilities.action = true;
        return info;
    }

    std::shared_ptr<Counters> counters_;
    bool return_null_session_;
};

wam::Model * make_model(const std::shared_ptr<Counters> & counters,
                        bool return_null_session = false) {
    return wam::internal::adopt_model(
        std::make_unique<FakeModel>(counters, return_null_session));
}

void test_deferred_model_destruction() {
    using wam::test::require;
    using wam::test::require_error;

    const auto counters = std::make_shared<Counters>();
    wam::Model * model = make_model(counters);
    require(wam::model_info(model).architecture == "test-only",
            "model_info did not delegate to ModelImpl");

    wam::Session * first = wam::session_create(model);
    wam::Session * second = wam::session_create(model);
    require(counters->sessions_created == 2, "session creation count changed");

    const wam::Prediction prediction = wam::predict(first, {});
    require(prediction.action.data.size() == 2,
            "predict did not delegate to SessionImpl");

    wam::model_free(model);
    require(counters->models_destroyed == 0,
            "model resources were destroyed while sessions remained");
    require_error([&] { wam::session_create(model); },
                  wam::ErrorCode::invalid_argument,
                  "released model must reject new sessions");
    require_error([&] { (void) wam::model_info(model); },
                  wam::ErrorCode::invalid_argument,
                  "released model must reject model_info");

    const wam::Prediction after_release = wam::predict(second, {});
    require(after_release.action.data.size() == 2,
            "existing session stopped after model handle release");

    wam::session_free(first);
    require(counters->models_destroyed == 0,
            "model resources were destroyed before the last session");
    wam::session_free(second);
    require(counters->models_destroyed == 1,
            "model resources were not destroyed with the last session");
    require(counters->sessions_destroyed == 2,
            "session implementations were not destroyed exactly once");
}

void test_reset_translation() {
    using wam::test::require;

    const auto counters = std::make_shared<Counters>();
    wam::Model * model = make_model(counters);

    wam::Session * success = wam::session_create(model, {});
    require(static_cast<bool>(wam::session_reset(success)),
            "successful reset was not preserved");

    wam::SessionOptions wam_error_options;
    wam_error_options.random_seed = 1;
    wam::Session * wam_error = wam::session_create(model, wam_error_options);
    const wam::Status wam_status = wam::session_reset(wam_error);
    require(wam_status.code == wam::ErrorCode::failed_precondition,
            "wam::Error reset code was not preserved");
    require(wam_status.details.size() == 1,
            "wam::Error reset details were not preserved");

    wam::SessionOptions standard_error_options;
    standard_error_options.random_seed = 2;
    wam::Session * standard_error =
        wam::session_create(model, standard_error_options);
    require(wam::session_reset(standard_error).code == wam::ErrorCode::internal,
            "std::exception reset must map to internal");

    wam::SessionOptions unknown_error_options;
    unknown_error_options.random_seed = 3;
    wam::Session * unknown_error =
        wam::session_create(model, unknown_error_options);
    require(wam::session_reset(unknown_error).code == wam::ErrorCode::internal,
            "unknown reset exception must map to internal");

    wam::session_free(success);
    wam::session_free(wam_error);
    wam::session_free(standard_error);
    wam::session_free(unknown_error);
    wam::model_free(model);
    require(counters->models_destroyed == 1,
            "normal model destruction failed");
}

void test_invalid_handles_and_options() {
    using wam::test::require;
    using wam::test::require_error;

    wam::model_free(nullptr);
    wam::session_free(nullptr);
    require(wam::session_reset(nullptr).code == wam::ErrorCode::invalid_argument,
            "null reset must return invalid_argument");
    require_error([&] { (void) wam::session_create(nullptr); },
                  wam::ErrorCode::invalid_argument, "null model handle");
    require_error([&] { (void) wam::predict(nullptr, {}); },
                  wam::ErrorCode::invalid_argument, "null session handle");
    require_error([&] { (void) wam::model_info(nullptr); },
                  wam::ErrorCode::invalid_argument, "null model_info handle");
    require_error(
        [&] { (void) wam::internal::adopt_model(nullptr); },
        wam::ErrorCode::internal, "null ModelImpl adoption");

    const auto counters = std::make_shared<Counters>();
    wam::Model * null_session_model = make_model(counters, true);
    require_error([&] { (void) wam::session_create(null_session_model); },
                  wam::ErrorCode::internal, "null SessionImpl result");
    wam::model_free(null_session_model);

    wam::ModelOptions options;
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::invalid_argument, "empty artifact path");

    options.artifact_path = "not-opened-in-slice-1.gguf";
    options.backend = wam::Backend::unknown;
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::invalid_argument, "unknown backend");

    options.backend = wam::Backend::automatic;
    options.compute_precision = wam::ComputePrecision::unknown;
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::invalid_argument, "unknown precision");

    options.compute_precision = wam::ComputePrecision::automatic;
    options.device_index = -1;
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::invalid_argument, "negative device index");

    options.device_index = 0;
    options.fixed_prompt = wam::FixedPrompt{{1, 2}, {1}};
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::invalid_argument,
                  "fixed prompt size mismatch");

    options.fixed_prompt.reset();
    require_error([&] { (void) wam::model_load(options); },
                  wam::ErrorCode::not_found,
                  "valid options must proceed to artifact open");
}

} // namespace

int main() {
    test_deferred_model_destruction();
    test_reset_translation();
    test_invalid_handles_and_options();
    return 0;
}
