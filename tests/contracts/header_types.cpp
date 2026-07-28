#include "wam/types.h"

int main() {
    return wam::dtype_size(wam::DType::f32) == 4 ? 0 : 1;
}
