#include "wam/version.h"

static_assert(WAM_VERSION_MAJOR == 0, "unexpected major version");
static_assert(WAM_VERSION_MINOR == 6, "unexpected minor version");

int main() {
    return WAM_VERSION_PATCH;
}
