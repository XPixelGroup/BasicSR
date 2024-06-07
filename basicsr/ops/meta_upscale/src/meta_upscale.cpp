#include <torch/extension.h>

#include <vector>

torch::Tensor meta_upscale_forward(
    torch::Tensor x,
    torch::Tensor weight,
    float s_v,
    float s_h,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size);

std::vector<torch::Tensor> meta_upscale_backward(
    torch::Tensor dout,
    torch::Tensor x,
    torch::Tensor weight,
    float s_v,
    float s_h,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &meta_upscale_forward, "meta upscale forward (CUDA)");
    m.def("backward", &meta_upscale_backward, "meta upscale backward (CUDA)");
}
