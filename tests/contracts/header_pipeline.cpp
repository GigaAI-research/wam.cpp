#include "wam/pipeline.h"

#include <type_traits>

static_assert(!std::is_copy_constructible<wam::Pipeline>::value,
              "Pipeline must be move-only");

int main() {
    return 0;
}
