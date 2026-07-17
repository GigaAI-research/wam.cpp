#pragma once

namespace wam::internal::gwp05::engine {

struct ImageView;
struct Gwp05CudaPreprocessor;

Gwp05CudaPreprocessor * gwp05_cuda_preprocessor_create();
void gwp05_cuda_preprocessor_destroy(Gwp05CudaPreprocessor * preprocessor);

bool gwp05_cuda_preprocess(Gwp05CudaPreprocessor * preprocessor,
                           const ImageView * images, int image_count,
                           int output_width, int output_height,
                           void * output_device);

} // namespace wam::internal::gwp05::engine
