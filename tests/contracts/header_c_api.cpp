#include "wam/c_api.h"

static_assert(WAM_C_ABI_VERSION == 4U, "Phase 7 requires C ABI v4");
static_assert(WAM_C_STRUCT_VERSION == 1U,
              "unexpected C ABI struct version");

int main() {
    wam_c_model_options options{};
    wam_c_model * model = nullptr;
    return model == nullptr && options.struct_version == 0U ? 0 : 1;
}
