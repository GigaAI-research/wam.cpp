#include "wam/wam.h"
#include "wam/c_api.h"

#include <type_traits>

static_assert(WAM_VERSION_MAJOR == 0, "unexpected major version");
static_assert(WAM_VERSION_MINOR == 6, "unexpected minor version");
static_assert(std::is_destructible<wam::ModelOptions>::value,
              "public ModelOptions must be a complete type");
static_assert(std::is_destructible<wam::Prediction>::value,
              "public Prediction must be a complete type");
static_assert(sizeof(wam_c_model_options) > 0,
              "serving C ABI options must be a complete C type");
static_assert(WAM_C_ABI_VERSION == 3U, "unexpected serving C ABI version");

int main() {
    wam::Model * model = nullptr;
    wam::Session * session = nullptr;
    wam::model_free(model);
    wam::session_free(session);
    return wam::Status::success() ? 0 : 1;
}
