#include "cuda_preprocess.h"
#include "models/gwp05/engine/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace wam::internal::gwp05::engine {

struct Gwp05CudaPreprocessor {
    void * images[3]{};
    size_t capacities[3]{};
};

namespace {

bool cuda_ok(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "wam(gwp05): CUDA image preprocessing %s failed: %s\n",
                 operation, cudaGetErrorString(status));
    return false;
}

__device__ float read_pixel(const void * source, int format, int width,
                            int x, int y, int channel) {
    const size_t index = (static_cast<size_t>(y) * width + x) * 3 + channel;
    return format == static_cast<int>(PixelFormat::U8)
        ? static_cast<const std::uint8_t *>(source)[index] * (1.0f / 255.0f)
        : static_cast<const float *>(source)[index];
}

__device__ void axis_parameters(int output_position, int source_size,
                                int resized_size, int destination_size,
                                int & first, int & last,
                                float & center, float & filter_scale) {
    const int crop = (resized_size - destination_size) / 2;
    const float axis_scale = static_cast<float>(resized_size) / source_size;
    filter_scale = fmaxf(1.0f, 1.0f / axis_scale);
    center = (output_position + crop + 0.5f) / axis_scale - 0.5f;
    first = static_cast<int>(ceilf(center - filter_scale));
    last = static_cast<int>(floorf(center + filter_scale));
}

__global__ void preprocess_kernel(
        const void * image0, const void * image1, const void * image2,
        int format0, int format1, int format2,
        int width0, int height0, int width1, int height1,
        int width2, int height2, int output_width, int output_height,
        float * output) {
    const int patch_width = output_width / 2;
    const int patch_height = output_height / 2;
    const size_t elements = static_cast<size_t>(12) * patch_width * patch_height;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const int x = index % patch_width;
    const int y = (index / patch_width) % patch_height;
    const int packed_channel = index /
        (static_cast<size_t>(patch_width) * patch_height);
    const int source_channel = packed_channel / 4;
    const int dx = (packed_channel % 4) / 2;
    const int dy = packed_channel % 2;
    const int canvas_x = x * 2 + dx;
    const int canvas_y = y * 2 + dy;

    const int top_height = output_height / 2;
    const int left_width = output_width / 2;
    const void * source;
    int format, source_width, source_height, destination_width, destination_height;
    int local_x, local_y;
    if (canvas_y < top_height) {
        source = image0;
        format = format0;
        source_width = width0;
        source_height = height0;
        destination_width = output_width;
        destination_height = top_height;
        local_x = canvas_x;
        local_y = canvas_y;
    } else if (canvas_x < left_width) {
        source = image1;
        format = format1;
        source_width = width1;
        source_height = height1;
        destination_width = left_width;
        destination_height = output_height - top_height;
        local_x = canvas_x;
        local_y = canvas_y - top_height;
    } else {
        source = image2;
        format = format2;
        source_width = width2;
        source_height = height2;
        destination_width = output_width - left_width;
        destination_height = output_height - top_height;
        local_x = canvas_x - left_width;
        local_y = canvas_y - top_height;
    }

    int x_first, x_last, y_first, y_last;
    float x_center, x_filter, y_center, y_filter;
    const float resize_scale = fmaxf(
        static_cast<float>(destination_width) / source_width,
        static_cast<float>(destination_height) / source_height);
    const int resized_width = static_cast<int>(roundf(source_width * resize_scale));
    const int resized_height = static_cast<int>(roundf(source_height * resize_scale));
    axis_parameters(local_x, source_width, resized_width, destination_width,
                    x_first, x_last, x_center, x_filter);
    axis_parameters(local_y, source_height, resized_height, destination_height,
                    y_first, y_last, y_center, y_filter);

    float x_total = 0.0f;
    for (int sx = x_first; sx <= x_last; ++sx) {
        x_total += fmaxf(0.0f, 1.0f - fabsf(sx - x_center) / x_filter);
    }
    float y_total = 0.0f;
    for (int sy = y_first; sy <= y_last; ++sy) {
        y_total += fmaxf(0.0f, 1.0f - fabsf(sy - y_center) / y_filter);
    }

    float value = 0.0f;
    for (int sy = y_first; sy <= y_last; ++sy) {
        const float wy = fmaxf(0.0f, 1.0f - fabsf(sy - y_center) / y_filter) / y_total;
        const int clamped_y = max(0, min(source_height - 1, sy));
        float horizontal = 0.0f;
        for (int sx = x_first; sx <= x_last; ++sx) {
            const float wx = fmaxf(0.0f, 1.0f - fabsf(sx - x_center) / x_filter) /
                x_total;
            const int clamped_x = max(0, min(source_width - 1, sx));
            horizontal += read_pixel(source, format, source_width, clamped_x,
                                     clamped_y, source_channel) * wx;
        }
        value += horizontal * wy;
    }
    output[index] = value * 2.0f - 1.0f;
}

} // namespace

Gwp05CudaPreprocessor * gwp05_cuda_preprocessor_create() {
    return new Gwp05CudaPreprocessor();
}

void gwp05_cuda_preprocessor_destroy(Gwp05CudaPreprocessor * preprocessor) {
    if (!preprocessor) return;
    for (void * image : preprocessor->images) cudaFree(image);
    delete preprocessor;
}

bool gwp05_cuda_preprocess(
        Gwp05CudaPreprocessor * preprocessor, const ImageView * images,
        int image_count, int output_width, int output_height, void * output_device) {
    if (!preprocessor || !images || image_count != 3 || !output_device ||
        output_width <= 0 || output_height <= 0) {
        return false;
    }
    for (int index = 0; index < 3; ++index) {
        const ImageView & image = images[index];
        if (!image.data || image.w <= 0 || image.h <= 0 ||
            (image.format != PixelFormat::U8 &&
             image.format != PixelFormat::F32_RGB_01)) {
            return false;
        }
        const size_t element_size = image.format == PixelFormat::U8
            ? sizeof(std::uint8_t) : sizeof(float);
        const size_t bytes = static_cast<size_t>(image.w) * image.h * 3 * element_size;
        if (bytes > preprocessor->capacities[index]) {
            if (preprocessor->images[index] &&
                !cuda_ok(cudaFree(preprocessor->images[index]), "buffer release")) {
                return false;
            }
            preprocessor->images[index] = nullptr;
            if (!cuda_ok(cudaMalloc(&preprocessor->images[index], bytes),
                         "buffer allocation")) {
                return false;
            }
            preprocessor->capacities[index] = bytes;
        }
        if (!cuda_ok(cudaMemcpyAsync(preprocessor->images[index], image.data, bytes,
                                     cudaMemcpyHostToDevice), "image upload")) {
            return false;
        }
    }

    const size_t elements = static_cast<size_t>(12) * (output_width / 2) *
        (output_height / 2);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    preprocess_kernel<<<blocks, threads>>>(
        preprocessor->images[0], preprocessor->images[1], preprocessor->images[2],
        static_cast<int>(images[0].format), static_cast<int>(images[1].format),
        static_cast<int>(images[2].format), images[0].w, images[0].h,
        images[1].w, images[1].h, images[2].w, images[2].h,
        output_width, output_height, static_cast<float *>(output_device));
    if (!cuda_ok(cudaGetLastError(), "kernel launch")) return false;
    return cuda_ok(cudaDeviceSynchronize(), "synchronization");
}

} // namespace wam::internal::gwp05::engine
