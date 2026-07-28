#include "wam/session.h"

#include <type_traits>

static_assert(!std::is_copy_constructible<wam::Session>::value,
              "Session must be move-only");

int main() {
    return 0;
}
