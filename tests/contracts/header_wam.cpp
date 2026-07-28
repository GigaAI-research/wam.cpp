#include "wam/wam.h"

int main() {
    return wam::dtype_size(wam::DType::f32) == sizeof(float) ? 0 : 1;
}
