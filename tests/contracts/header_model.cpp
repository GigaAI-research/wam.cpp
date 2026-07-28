#include "wam/model.h"

#include <type_traits>

static_assert(!std::is_copy_constructible<wam::Model>::value,
              "Model must be move-only");

int main() {
    return 0;
}
