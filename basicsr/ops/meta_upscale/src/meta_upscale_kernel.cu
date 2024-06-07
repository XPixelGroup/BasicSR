#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 16
#define WARP_SIZE 32

// // x: (n, c_in, h_in, w_in)
// // weight: (out_h * out_w, k * k * in_c * out_c)
// // out: (n, c_out, h_out, w_out)
// __global__ void meta_upscale_forward_kernel(float *x, float *weight, float *out,
//     float s_v, float s_h,
//     int n, int c_in, int h_in, int w_in, int c_out, int h_out, int w_out)
// {
//     const int tid_h = threadIdx.y;
//     const int tid_w = threadIdx.x;
//     const int gid_h = threadIdx.y + blockIdx.y * blockDim.y;
//     const int gid_w = threadIdx.x + blockIdx.x * blockDim.x;

//     if (gid_h >= h_out || gid_w >= w_out) return;

//     const int i_prime = (float)h_out / s_v;
//     const int j_prime = (float)w_out / s_h;

//     __shared__ float sw[BLOCK_SIZE][BLOCK_SIZE][KERNEL_SIZE * KERNEL_SIZE * C_IN * C_OUT];
//     __shared__ float sx[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1][BATCH_SIZE * C_IN];

//     for (int co = 0; co < c_out; ++co)
//         for (int ci = 0; ci < c_in; ++ci)
//             for (int k1 = 0; k1 < KERNEL_SIZE; ++k1)
//                 for (int k2 = 0; k2 < KERNEL_SIZE; ++k2)
//                 {
//                     int sw_idx = k2 + k1 * KERNEL_SIZE + ci * KERNEL_SIZE * KERNEL_SIZE + co * KERNEL_SIZE * KERNEL_SIZE * c_in; // [co][ci][k1][k2]
//                     int weight_idx = sw_idx + gid_w * KERNEL_SIZE * KERNEL_SIZE * c_in * c_out + gid_h * KERNEL_SIZE * KERNEL_SIZE * c_in * c_out * w_out;
//                     sw[tid_h][tid_w][sw_idx] = weight[weight_idx];
//                 }
    
//     for (int ibatch = 0; ibatch < n; ++ibatch)
//         for (int ci = 0; ci < c_in; ++ci)
//         {
//             int sx_idx = ci + n * c_in;
//             int x_idx = j_prime + i_prime * w_out + ci * w_out * h_out + ibatch * w_out * h_out * c_in;
//             sx[tid_h][tid_w][sx_idx] = x[x_idx];
//             if (tid_h < KERNEL_SIZE - 1 && tid_w < KERNEL_SIZE - 1 && i_prime + BLOCK_SIZE < h_in && j_prime + BLOCK_SIZE < w_out)
//             {
//                 sx[tid_h + BLOCK_SIZE][tid_w + BLOCK_SIZE][sx_idx] = x[x_idx + BLOCK_SIZE + BLOCK_SIZE * w_out];
//             }

//         }

//     __syncthreads();
    

//     for (int ibatch = 0; ibatch < n; ++ibatch)
//         for (int co = 0; co < c_out; ++co)
//         {
//             float sum = 0.0;
//             for (int ci = 0; ci < c_in; ++ci)
//                 for (int k1 = 0; k1 < KERNEL_SIZE; ++k1)
//                     for (int k2 = 0; k2 < KERNEL_SIZE; ++k2)
//                     {
//                         int sw_idx = k2 + k1 * KERNEL_SIZE + ci * KERNEL_SIZE * KERNEL_SIZE + co * KERNEL_SIZE * KERNEL_SIZE * c_in;
//                         int sx_idx = ci + n * c_in;
//                         sum += sw[tid_h][tid_w][sw_idx] * sx[tid_h + k1][tid_h + k2][sx_idx];
//                     }
//             int out_idx = gid_w + gid_h * h_out + co * h_out * w_out + ibatch * h_out * w_out * c_out;
//             out[out_idx] = sum;
//         }
// }

// x: (n, c_in + pad, h_in + pad, w_in)
// weight: (out_h * out_w, k * k * in_c * out_c)
// out: (n, c_out, h_out, w_out)
__global__ void meta_upscale_forward_kernel(float *x, float *weight, float *out,
    float s_v, float s_h,
    int n, int c_in, int h_in, int w_in, int c_out, int h_out, int w_out, int ks)
{
    // const int tid_h = threadIdx.y;
    // const int tid_w = threadIdx.x;
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= h_out || j >= w_out) return;

    int i_p = i / s_v;
    int j_p = j / s_h;
    int h_in_pad = h_in + ks - 1;
    int w_in_pad = w_in + ks - 1;
    
    for (int ibatch = 0; ibatch < n; ++ibatch)
        for (int k1 = 0; k1 < ks; ++k1)
            for (int k2 = 0; k2 < ks; ++k2)
                for (int ci = 0; ci < c_in; ++ci)
                    for (int co = 0; co < c_out; ++co)
                    {
                        // w: (h_out, w_out, ks, ks, c_in, c_out)
                        // x: (n, c_in, h_in + pad, w_in + pad)

                        // w[i][j][k1][k2][ci][co]
                        int w_idx = co + ci * (c_out) + k2 * (c_out * c_in) \
                            + k1 * (c_out * c_in * ks) + j * (c_out * c_in * ks * ks) \
                            + i * (c_out * c_in * ks * ks * w_out);
                        // x[ibatch][ci][i_p + k1][j_p + k2]
                        int x_idx = (j_p + k2) + (i_p + k1) * (w_in_pad) + ci * (w_in_pad * h_in_pad) + ibatch * (w_in_pad * h_in_pad * c_in);
                        // out[ibatch][co][i][j]
                        int out_idx = j + i * (w_out) + co * (w_out * h_out) + ibatch * (w_out * h_out * c_out);
                        out[out_idx] += weight[w_idx] * x[x_idx];
                    }
}

__global__ void meta_upscale_backward_kernel(float *dx, float *dweight, float *dout, float *x, float *weight,
    float s_v, float s_h,
    int n, int c_in, int h_in, int w_in, int c_out, int h_out, int w_out, int ks)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= h_out || j >= w_out) return;

    int i_p = i / s_v;
    int j_p = j / s_h;
    int h_in_pad = h_in + ks - 1;
    int w_in_pad = w_in + ks - 1;
    
    for (int ibatch = 0; ibatch < n; ++ibatch)
        for (int k1 = 0; k1 < ks; ++k1)
            for (int k2 = 0; k2 < ks; ++k2)
                for (int ci = 0; ci < c_in; ++ci)
                    for (int co = 0; co < c_out; ++co)
                    {
                        int w_idx = co + ci * (c_out) + k2 * (c_out * c_in) \
                            + k1 * (c_out * c_in * ks) + j * (c_out * c_in * ks * ks) \
                            + i * (c_out * c_in * ks * ks * w_out);
                        int x_idx = (j_p + k2) + (i_p + k1) * (w_in_pad) + ci * (w_in_pad * h_in_pad) + ibatch * (w_in_pad * h_in_pad * c_in);
                        int out_idx = j + i * (w_out) + co * (w_out * h_out) + ibatch * (w_out * h_out * c_out);
                        // Calculate gradients wrt x and weight
                        dweight[w_idx] += x[x_idx] * dout[out_idx];
                        atomicAdd(&dx[x_idx], weight[w_idx] * dout[out_idx]);
                    }

}

torch::Tensor meta_upscale_forward(
    torch::Tensor x, torch::Tensor weight, float s_v, float s_h,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width, int kernel_size)
{
    // int block_size = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int block_size_x = BLOCK_SIZE * 2;
    int grid_size_y = (out_height - 1) / block_size_y + 1;
    int grid_size_x = (out_width - 1) / block_size_x + 1;
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);

    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    auto out = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // printf("grid: (%d, %d), block: (%d, %d)\n", grid_size_x, grid_size_y, block_size_x, block_size_y);

    meta_upscale_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), s_v, s_h,
        batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size
    );

    return out;
}

std::vector<torch::Tensor> meta_upscale_backward(
    torch::Tensor dout, torch::Tensor x, torch::Tensor weight,
    float s_v, float s_h, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width, int kernel_size)
{
    int block_size_y = BLOCK_SIZE;
    int block_size_x = BLOCK_SIZE * 2;
    int grid_size_y = (out_height - 1) / block_size_y + 1;
    int grid_size_x = (out_width - 1) / block_size_x + 1;
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);

    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    auto dx = torch::zeros_like(x);
    auto dweight = torch::zeros_like(weight);

    meta_upscale_backward_kernel<<<grid, block>>>(
        dx.data_ptr<float>(), dweight.data_ptr<float>(), dout.data_ptr<float>(),
        x.data_ptr<float>(), weight.data_ptr<float>(), s_v, s_h,
        batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size
    );

    return {dx, dweight};
}