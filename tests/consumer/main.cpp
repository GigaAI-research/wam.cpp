#include <wam/wam.h>

#include <type_traits>

static_assert(std::is_move_constructible<wam::Pipeline>::value,
              "installed Pipeline must be complete and movable");

int main() {
    return wam::dtype_size(wam::DType::f32) == sizeof(float) ? 0 : 1;
}
