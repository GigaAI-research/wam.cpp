#include "wam/c_api.h"
#include "wam/wam.h"

#include <type_traits>

static_assert(WAM_VERSION_MAJOR == 0, "unexpected major version");
static_assert(WAM_VERSION_MINOR == 6, "unexpected minor version");
static_assert(std::is_destructible<wam::RuntimeConfig>::value,
              "RuntimeConfig must be complete");
static_assert(std::is_destructible<wam::Prediction>::value,
              "Prediction must be complete");
static_assert(std::is_move_constructible<wam::Model>::value,
              "Model must be movable");
static_assert(!std::is_copy_constructible<wam::Model>::value,
              "Model must not be copyable");
static_assert(std::is_move_constructible<wam::Session>::value,
              "Session must be movable");
static_assert(!std::is_copy_constructible<wam::Session>::value,
              "Session must not be copyable");
static_assert(sizeof(wam_c_model_options) > 0,
              "serving C ABI options must be complete");
static_assert(WAM_C_ABI_VERSION == 4U, "unexpected serving C ABI version");

int main() {
    wam::Observation observation;
    wam::Prediction prediction;
    return observation.images.empty() && prediction.action.empty() ? 0 : 1;
}
