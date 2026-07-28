#include "fixtures/fake_public_model.h"
#include "support/test_utils.h"

#include "wam/error.h"
#include "wam/model.h"
#include "wam/observation.h"
#include "wam/session.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace {

static_assert(!std::is_copy_constructible<wam::Model>::value,
              "Model must remain move-only");
static_assert(!std::is_copy_constructible<wam::Session>::value,
              "Session must remain move-only");
static_assert(std::is_nothrow_move_constructible<wam::Model>::value,
              "Model moves must not throw");
static_assert(std::is_nothrow_move_constructible<wam::Session>::value,
              "Session moves must not throw");

void test_session_keeps_model_alive() {
    using wam::test::require;
    using wam::test::require_error;

    const auto counters = std::make_shared<wam::test::FakeModelCounters>();
    wam::Session surviving_session = [&] {
        wam::Model model = wam::test::make_fake_model(counters);
        require(model.info().architecture == "test-only",
                "Model::info did not delegate to the implementation");
        require(model.info().policy_spec != nullptr &&
                    model.info().policy_spec->identity.profile == "fake-v1",
                "ModelInfo must expose an immutable PolicySpec");

        wam::Session first = model.create_session();
        wam::Session second = model.create_session();
        require(counters->sessions_created == 2,
                "session creation count changed");
        require(first.predict({}).action.data.size() == 2,
                "Session::predict did not delegate to the implementation");

        wam::Session moved = std::move(first);
        require_error([&] { (void) first.predict({}); },
                      wam::ErrorCode::failed_precondition,
                      "moved-from Session must reject predict");
        require(moved.predict({}).action.horizon() == 1,
                "moved Session lost its implementation");
        return second;
    }();

    require(counters->models_destroyed == 0,
            "Session did not retain shared model resources");
    require(surviving_session.predict({}).action.action_dimension() == 2,
            "Session stopped working after Model destruction");
}

void test_move_and_reset_errors() {
    using wam::test::require;
    using wam::test::require_error;

    const auto counters = std::make_shared<wam::test::FakeModelCounters>();
    wam::Model model = wam::test::make_fake_model(counters);
    wam::Model moved_model = std::move(model);
    require_error([&] { (void) model.info(); },
                  wam::ErrorCode::failed_precondition,
                  "moved-from Model must reject info");
    require_error([&] { (void) model.create_session(); },
                  wam::ErrorCode::failed_precondition,
                  "moved-from Model must reject create_session");

    wam::Session success = moved_model.create_session();
    success.reset();

    wam::SessionConfig failing_config;
    failing_config.random_seed = 1;
    wam::Session failing = moved_model.create_session(failing_config);
    bool saw_structured_error = false;
    try {
        failing.reset();
    } catch (const wam::Error & error) {
        require(error.code() == wam::ErrorCode::failed_precondition,
                "reset changed the implementation error code");
        require(error.details().size() == 1,
                "reset changed structured error details");
        saw_structured_error = true;
    }
    require(saw_structured_error, "reset did not propagate wam::Error");

    wam::SessionConfig standard_config;
    standard_config.random_seed = 2;
    wam::Session standard_failure =
        moved_model.create_session(standard_config);
    require_error([&] { standard_failure.reset(); },
                  wam::ErrorCode::internal,
                  "std::exception must be translated at the public boundary");

    wam::SessionConfig unknown_config;
    unknown_config.random_seed = 3;
    wam::Session unknown_failure = moved_model.create_session(unknown_config);
    require_error([&] { unknown_failure.reset(); }, wam::ErrorCode::internal,
                  "unknown exception must be translated at the public boundary");
}

void test_invalid_construction_and_config() {
    using wam::test::require_error;

    const auto counters = std::make_shared<wam::test::FakeModelCounters>();
    wam::Model null_session_model =
        wam::test::make_fake_model(counters, true);
    require_error([&] { (void) null_session_model.create_session(); },
                  wam::ErrorCode::internal,
                  "null SessionImpl result must be rejected");

    require_error([&] { (void) wam::Model::load(""); },
                  wam::ErrorCode::invalid_argument, "empty artifact path");

    wam::RuntimeConfig config;
    config.backend = wam::Backend::unknown;
    require_error([&] { (void) wam::Model::load("missing.gguf", config); },
                  wam::ErrorCode::invalid_argument, "unknown backend");

    config.backend = wam::Backend::automatic;
    config.compute_precision = wam::ComputePrecision::unknown;
    require_error([&] { (void) wam::Model::load("missing.gguf", config); },
                  wam::ErrorCode::invalid_argument, "unknown precision");

    config.compute_precision = wam::ComputePrecision::automatic;
    config.device_index = -1;
    require_error([&] { (void) wam::Model::load("missing.gguf", config); },
                  wam::ErrorCode::invalid_argument, "negative device index");

    config.device_index = 0;
    config.fixed_prompt = wam::FixedPrompt{{1, 2}, {1}};
    require_error([&] { (void) wam::Model::load("missing.gguf", config); },
                  wam::ErrorCode::invalid_argument,
                  "fixed prompt size mismatch");

    config.fixed_prompt.reset();
    require_error([&] { (void) wam::Model::load("missing.gguf", config); },
                  wam::ErrorCode::not_found,
                  "valid config must proceed to artifact open");
}

} // namespace

int main() {
    test_session_keeps_model_alive();
    test_move_and_reset_errors();
    test_invalid_construction_and_config();
    return 0;
}
