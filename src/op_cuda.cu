#include "op_cuda.h"
#include <cmath>

// A helper for launching kernels
static inline int getGridSize(int n, int block_size=256) {
    return (n + block_size - 1) / block_size;
}

// -------------------- Add --------------------
__global__ void add_forward_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

void add_forward_cuda(const float* a, const float* b, float* out, int size) {
    int grid = getGridSize(size);
    add_forward_kernel<<<grid, 256>>>(a, b, out, size);
    cudaDeviceSynchronize();
}

__global__ void add_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] += grad_out[idx];
        grad_b[idx] += grad_out[idx];
    }
}

void add_backward_cuda(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int grid = getGridSize(size);
    add_backward_kernel<<<grid, 256>>>(grad_out, grad_a, grad_b, size);
    cudaDeviceSynchronize();
}

// -------------------- Subtract --------------------
__global__ void sub_forward_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

void sub_forward_cuda(const float* a, const float* b, float* out, int size) {
    int grid = getGridSize(size);
    sub_forward_kernel<<<grid, 256>>>(a, b, out, size);
    cudaDeviceSynchronize();
}

__global__ void sub_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] += grad_out[idx];
        grad_b[idx] -= grad_out[idx];
    }
}

void sub_backward_cuda(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int grid = getGridSize(size);
    sub_backward_kernel<<<grid, 256>>>(grad_out, grad_a, grad_b, size);
    cudaDeviceSynchronize();
}

// -------------------- Multiply --------------------
__global__ void mul_forward_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

void mul_forward_cuda(const float* a, const float* b, float* out, int size) {
    int grid = getGridSize(size);
    mul_forward_kernel<<<grid, 256>>>(a, b, out, size);
    cudaDeviceSynchronize();
}

__global__ void mul_backward_kernel(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] += grad_out[idx] * b[idx];
        grad_b[idx] += grad_out[idx] * a[idx];
    }
}

void mul_backward_cuda(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int grid = getGridSize(size);
    mul_backward_kernel<<<grid, 256>>>(grad_out, a, b, grad_a, grad_b, size);
    cudaDeviceSynchronize();
}

// -------------------- Divide --------------------
__global__ void div_forward_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float denom = b[idx];
        // Assume no zero division as checked in CPU code
        out[idx] = a[idx] / denom;
    }
}

void div_forward_cuda(const float* a, const float* b, float* out, int size) {
    int grid = getGridSize(size);
    div_forward_kernel<<<grid, 256>>>(a, b, out, size);
    cudaDeviceSynchronize();
}

__global__ void div_backward_kernel(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float A = a[idx];
        float B = b[idx];
        grad_a[idx] += grad_out[idx] / B;
        grad_b[idx] -= (A * grad_out[idx]) / (B * B);
    }
}

void div_backward_cuda(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size) {
    int grid = getGridSize(size);
    div_backward_kernel<<<grid, 256>>>(grad_out, a, b, grad_a, grad_b, size);
    cudaDeviceSynchronize();
}

// -------------------- Exp --------------------
__global__ void exp_forward_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = expf(a[idx]);
    }
}

void exp_forward_cuda(const float* a, float* out, int size) {
    int grid = getGridSize(size);
    exp_forward_kernel<<<grid, 256>>>(a, out, size);
    cudaDeviceSynchronize();
}

__global__ void exp_backward_kernel(const float* grad_out, const float* a, float* grad_a, int size) {
    // Need to recompute exp(a)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = expf(a[idx]);
        grad_a[idx] += grad_out[idx] * val;
    }
}

void exp_backward_cuda(const float* grad_out, const float* a, float* grad_a, int size) {
    int grid = getGridSize(size);
    exp_backward_kernel<<<grid, 256>>>(grad_out, a, grad_a, size);
    cudaDeviceSynchronize();
}

// -------------------- Tanh --------------------
// We'll assume we have 'out' as the forward output passed to backward for simplicity.
// If not, you'd need to recompute out = tanh(a[idx]) in backward.
__global__ void tanh_forward_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(a[idx]);
    }
}

void tanh_forward_cuda(const float* a, float* out, int size) {
    int grid = getGridSize(size);
    tanh_forward_kernel<<<grid, 256>>>(a, out, size);
    cudaDeviceSynchronize();
}

__global__ void tanh_backward_kernel(const float* grad_out, const float* out, float* grad_a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = out[idx];
        grad_a[idx] += grad_out[idx] * (1.0f - t * t);
    }
}

void tanh_backward_cuda(const float* grad_out, const float* out, float* grad_a, int size) {
    int grid = getGridSize(size);
    tanh_backward_kernel<<<grid, 256>>>(grad_out, out, grad_a, size);
    cudaDeviceSynchronize();
}

// -------------------- Relu --------------------
__global__ void relu_forward_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] > 0.0f) ? a[idx] : 0.0f;
    }
}

void relu_forward_cuda(const float* a, float* out, int size) {
    int grid = getGridSize(size);
    relu_forward_kernel<<<grid, 256>>>(a, out, size);
    cudaDeviceSynchronize();
}

__global__ void relu_backward_kernel(const float* grad_out, const float* a, float* grad_a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] += (a[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

void relu_backward_cuda(const float* grad_out, const float* a, float* grad_a, int size) {
    int grid = getGridSize(size);
    relu_backward_kernel<<<grid, 256>>>(grad_out, a, grad_a, size);
    cudaDeviceSynchronize();
}

// -------------------- Sum --------------------
// forward: sum all elements of a into out[0]
__global__ void sum_forward_kernel(const float* a, float* out, int size) {
    // We'll do a simple parallel reduction. For simplicity, we might do a naive approach.
    // In real code, use a proper reduction for performance.
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < size) val = a[idx];
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // block 0 accumulates into out[0]
    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void sum_forward_cuda(const float* a, float* out, int size) {
    // Initialize out[0] = 0
    cudaMemset(out, 0, sizeof(float));
    int grid = getGridSize(size);
    sum_forward_kernel<<<grid, 256>>>(a, out, size);
    cudaDeviceSynchronize();
}

// backward: grad_out is shape [1], spread to all elements of grad_a
__global__ void sum_backward_kernel(const float* grad_out, float* grad_a, int size) {
    float g = grad_out[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] += g;
    }
}

void sum_backward_cuda(const float* grad_out, float* grad_a, int size) {
    int grid = getGridSize(size);
    sum_backward_kernel<<<grid, 256>>>(grad_out, grad_a, size);
    cudaDeviceSynchronize();
}

// -------------------- Stack --------------------
// stack forward: inputs is array of pointers, each input shape [1], output shape [num_inputs]
// For simplicity: we assume inputs pointers are device pointers accessible here.
__global__ void stack_forward_kernel(const float** inputs, float* out, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each input is scalar. idx < num_inputs
    if (idx < num_inputs) {
        out[idx] = inputs[idx][0];
    }
}

void stack_forward_cuda(const float** inputs, float* out, int num_inputs) {
    int grid = getGridSize(num_inputs);
    stack_forward_kernel<<<grid, 256>>>(inputs, out, num_inputs);
    cudaDeviceSynchronize();
}

// stack backward: grad_out shape [num_inputs], distribute each grad_out[i] to grad_ins[i][0]
__global__ void stack_backward_kernel(const float* grad_out, float** grad_ins, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_inputs) {
        grad_ins[idx][0] += grad_out[idx];
    }
}

void stack_backward_cuda(const float* grad_out, float** grad_ins, int num_inputs) {
    int grid = getGridSize(num_inputs);
    stack_backward_kernel<<<grid, 256>>>(grad_out, grad_ins, num_inputs);
    cudaDeviceSynchronize();
}
