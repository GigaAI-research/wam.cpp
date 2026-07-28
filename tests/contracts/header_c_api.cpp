#include "wam/c_api.h"

static_assert(WAM_C_ABI_VERSION == 3U, "Phase 1 must preserve the 0.5 C ABI");

int main() {
    wam_c_model * model = nullptr;
    return model == nullptr ? 0 : 1;
}
