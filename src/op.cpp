#include <math.h>
#include <stdexcept>
#include "op.h"
#include "tensor.h"
#include "op_cuda.h"

static void check_same_shape_for_binary(const std::vector<std::shared_ptr<Tensor>> &inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("Binary op expected 2 inputs, got " + std::to_string(inputs.size()));
    }
    if (inputs[0]->shape != inputs[1]->shape)
    {
        throw std::invalid_argument("Binary op shapes must match.");
    }
}

static void check_one_input(const std::vector<std::shared_ptr<Tensor>> &inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Unary op expected 1 input, got " + std::to_string(inputs.size()));
    }
}

/////////////////// AddOp ///////////////////

void AddOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device; // assume same device

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        // CPU path
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = inputs[0]->data[i] + inputs[1]->data[i];
        }
    }
    else
    {
        // GPU path
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();
        output->allocate_memory_on_device();

        inputs[0]->copy_to_device();
        inputs[1]->copy_to_device();

        add_forward_cuda(inputs[0]->d_data, inputs[1]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void AddOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        // CPU
        for (int i = 0; i < sz; i++)
        {
            inputs[0]->grad[i] += output->grad[i];
            inputs[1]->grad[i] += output->grad[i];
        }
    }
    else
    {
        // GPU
        output->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();

        add_backward_cuda(output->d_grad, inputs[0]->d_grad, inputs[1]->d_grad, sz);
    }
}

/////////////////// SubtractOp ///////////////////

void SubtractOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = inputs[0]->data[i] - inputs[1]->data[i];
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();
        output->allocate_memory_on_device();

        inputs[0]->copy_to_device();
        inputs[1]->copy_to_device();

        sub_forward_cuda(inputs[0]->d_data, inputs[1]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void SubtractOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            inputs[0]->grad[i] += output->grad[i];
            inputs[1]->grad[i] -= output->grad[i];
        }
    }
    else
    {
        output->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();

        sub_backward_cuda(output->d_grad, inputs[0]->d_grad, inputs[1]->d_grad, sz);
    }
}

/////////////////// MultiplyOp ///////////////////

void MultiplyOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = inputs[0]->data[i] * inputs[1]->data[i];
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();
        output->allocate_memory_on_device();

        inputs[0]->copy_to_device();
        inputs[1]->copy_to_device();

        mul_forward_cuda(inputs[0]->d_data, inputs[1]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void MultiplyOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            inputs[0]->grad[i] += inputs[1]->data[i] * output->grad[i];
            inputs[1]->grad[i] += inputs[0]->data[i] * output->grad[i];
        }
    }
    else
    {
        output->copy_to_device();
        inputs[0]->copy_to_device(); // we need a and b data for backward
        inputs[1]->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();

        mul_backward_cuda(output->d_grad, inputs[0]->d_data, inputs[1]->d_data, inputs[0]->d_grad, inputs[1]->d_grad, sz);
    }
}

/////////////////// DivideOp ///////////////////

void DivideOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            if (inputs[1]->data[i] == 0.0f)
                throw std::domain_error("Division by zero");
            output->data[i] = inputs[0]->data[i] / inputs[1]->data[i];
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();
        output->allocate_memory_on_device();

        inputs[0]->copy_to_device();
        inputs[1]->copy_to_device();

        // Assume no zero division since CPU checked
        div_forward_cuda(inputs[0]->d_data, inputs[1]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void DivideOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            float a = inputs[0]->data[i];
            float b = inputs[1]->data[i];
            inputs[0]->grad[i] += output->grad[i] / b;
            inputs[1]->grad[i] -= (a * output->grad[i]) / (b * b);
        }
    }
    else
    {
        output->copy_to_device();
        inputs[0]->copy_to_device();
        inputs[1]->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        inputs[1]->allocate_memory_on_device();

        div_backward_cuda(output->d_grad, inputs[0]->d_data, inputs[1]->d_data, inputs[0]->d_grad, inputs[1]->d_grad, sz);
    }
}

/////////////////// ExpOp ///////////////////

void ExpOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = std::exp(inputs[0]->data[i]);
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        output->allocate_memory_on_device();
        inputs[0]->copy_to_device();
        exp_forward_cuda(inputs[0]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void ExpOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            inputs[0]->grad[i] += std::exp(inputs[0]->data[i]) * output->grad[i];
        }
    }
    else
    {
        output->copy_to_device();
        inputs[0]->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        exp_backward_cuda(output->d_grad, inputs[0]->d_data, inputs[0]->d_grad, sz);
    }
}

/////////////////// TanhOp ///////////////////

void TanhOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = std::tanh(inputs[0]->data[i]);
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        output->allocate_memory_on_device();
        inputs[0]->copy_to_device();

        tanh_forward_cuda(inputs[0]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void TanhOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            float t = std::tanh(inputs[0]->data[i]);
            inputs[0]->grad[i] += (1.0f - t * t) * output->grad[i];
        }
    }
    else
    {
        // We have the output->data which is tanh(a). Use it directly.
        output->copy_to_device();
        inputs[0]->allocate_memory_on_device();

        tanh_backward_cuda(output->d_grad, output->d_data, inputs[0]->d_grad, sz);
    }
}

/////////////////// ReluOp ///////////////////

void ReluOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    output->device = inputs[0]->device;

    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            output->data[i] = (inputs[0]->data[i] > 0.0f) ? inputs[0]->data[i] : 0.0f;
        }
    }
    else
    {
        inputs[0]->allocate_memory_on_device();
        output->allocate_memory_on_device();
        inputs[0]->copy_to_device();

        relu_forward_cuda(inputs[0]->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void ReluOp::backward()
{
    int sz = output->size();
    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < sz; i++)
        {
            inputs[0]->grad[i] += (inputs[0]->data[i] > 0.0f) ? output->grad[i] : 0.0f;
        }
    }
    else
    {
        output->copy_to_device();
        inputs[0]->copy_to_device();
        inputs[0]->allocate_memory_on_device();
        relu_backward_cuda(output->d_grad, inputs[0]->d_data, inputs[0]->d_grad, sz);
    }
}

/////////////////// SumOp ///////////////////

void SumOp::forward()
{
    check_one_input(inputs);
    auto in = inputs[0];
    output = std::make_shared<Tensor>(std::vector<int>{1});
    output->device = in->device;

    int sz = in->size();
    if (output->device == DeviceType::CPU)
    {
        float total = 0.0f;
        for (auto v : in->data)
            total += v;
        output->data[0] = total;
    }
    else
    {
        in->allocate_memory_on_device();
        output->allocate_memory_on_device();
        in->copy_to_device();
        sum_forward_cuda(in->d_data, output->d_data, sz);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void SumOp::backward()
{
    auto in = inputs[0];
    int sz = in->size();

    if (output->device == DeviceType::CPU)
    {
        float grad_val = output->grad[0];
        for (auto &g : in->grad)
        {
            g += grad_val;
        }
    }
    else
    {
        output->copy_to_device();
        in->allocate_memory_on_device();
        sum_backward_cuda(output->d_grad, in->d_grad, sz);
    }
}

/////////////////// StackOp ///////////////////
// StackOp: forward: combine multiple [1]-shaped inputs
// backward: distribute grads

void StackOp::forward()
{
    // Suppose each input is shape [1]. Output is [N]
    int N = (int)inputs.size();
    output = std::make_shared<Tensor>(std::vector<int>{N});
    output->device = inputs[0]->device; // assume all same device

    if (output->device == DeviceType::CPU)
    {
        for (int i = 0; i < N; i++)
        {
            output->data[i] = inputs[i]->data[0];
        }
    }
    else
    {
        // GPU
        // We need an array of pointers to d_data of inputs
        // Allocate on host first
        std::vector<float *> d_inputs(N);
        for (int i = 0; i < N; i++)
        {
            inputs[i]->allocate_memory_on_device();
            inputs[i]->copy_to_device();
            d_inputs[i] = inputs[i]->d_data;
        }

        // Copy this pointer array to GPU
        float **d_input_ptrs;
        cudaMalloc((void **)&d_input_ptrs, N * sizeof(float *));
        cudaMemcpy(d_input_ptrs, d_inputs.data(), N * sizeof(float *), cudaMemcpyHostToDevice);

        output->allocate_memory_on_device();

        stack_forward_cuda((const float **)d_input_ptrs, output->d_data, N);

        cudaFree(d_input_ptrs);
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void StackOp::backward()
{
    int N = (int)inputs.size();
    if (output->device == DeviceType::CPU)
    {
        float *g_out = output->grad.data();
        for (int i = 0; i < N; i++)
        {
            inputs[i]->grad[0] += g_out[i];
        }
    }
    else
    {
        // GPU
        output->copy_to_device();
        // We need grad pointers for each input
        std::vector<float *> d_grads_in(N);
        for (int i = 0; i < N; i++)
        {
            inputs[i]->allocate_memory_on_device();
            d_grads_in[i] = inputs[i]->d_grad;
        }

        float **d_grad_ptrs;
        cudaMalloc((void **)&d_grad_ptrs, N * sizeof(float *));
        cudaMemcpy(d_grad_ptrs, d_grads_in.data(), N * sizeof(float *), cudaMemcpyHostToDevice);

        stack_backward_cuda(output->d_grad, d_grad_ptrs, N);

        cudaFree(d_grad_ptrs);
    }
}
