#ifndef OP_CUDA_H
#define OP_CUDA_H

#include <cuda_runtime.h>

// Add
void add_forward_cuda(const float *a, const float *b, float *out, int size);
void add_backward_cuda(const float *grad_out, float *grad_a, float *grad_b, int size);

// Subtract
void sub_forward_cuda(const float *a, const float *b, float *out, int size);
void sub_backward_cuda(const float *grad_out, float *grad_a, float *grad_b, int size);

// Multiply
void mul_forward_cuda(const float *a, const float *b, float *out, int size);
void mul_backward_cuda(const float *grad_out, const float *a, const float *b, float *grad_a, float *grad_b, int size);

// Divide
void div_forward_cuda(const float *a, const float *b, float *out, int size);
void div_backward_cuda(const float *grad_out, const float *a, const float *b, float *grad_a, float *grad_b, int size);

// Exp
void exp_forward_cuda(const float *a, float *out, int size);
void exp_backward_cuda(const float *grad_out, const float *a, float *grad_a, int size);

// Tanh
void tanh_forward_cuda(const float *a, float *out, int size);
void tanh_backward_cuda(const float *grad_out, const float *out, float *grad_a, int size);
// Note: out is tanh(a) from forward. We can store it to avoid recomputing tanh.

// Relu
void relu_forward_cuda(const float *a, float *out, int size);
void relu_backward_cuda(const float *grad_out, const float *a, float *grad_a, int size);

// Sum
void sum_forward_cuda(const float *a, float *out, int size);
void sum_backward_cuda(const float *grad_out, float *grad_a, int size);

// Stack
void stack_forward_cuda(const float **inputs, float *out, int num_inputs);
// Each input is shape [1], out is shape [num_inputs]
void stack_backward_cuda(const float *grad_out, float **grad_ins, int num_inputs);

#endif // OP_CUDA_H
