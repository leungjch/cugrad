#include <math.h>
#include <stdexcept>
#include "op.h"
#include "tensor.h"

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

void AddOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = inputs[0]->data[i] + inputs[1]->data[i];
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void AddOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        inputs[0]->grad[i] += output->grad[i];
        inputs[1]->grad[i] += output->grad[i];
    }
}

void SubtractOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = inputs[0]->data[i] - inputs[1]->data[i];
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void SubtractOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        inputs[0]->grad[i] += output->grad[i];
        inputs[1]->grad[i] -= output->grad[i];
    }
}

void MultiplyOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = inputs[0]->data[i] * inputs[1]->data[i];
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void MultiplyOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        inputs[0]->grad[i] += inputs[1]->data[i] * output->grad[i];
        inputs[1]->grad[i] += inputs[0]->data[i] * output->grad[i];
    }
}

void DivideOp::forward()
{
    check_same_shape_for_binary(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        if (inputs[1]->data[i] == 0.0f)
        {
            throw std::domain_error("DivideOp::forward - Division by zero at index " + std::to_string(i));
        }
        output->data[i] = inputs[0]->data[i] / inputs[1]->data[i];
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void DivideOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        float a = inputs[0]->data[i];
        float b = inputs[1]->data[i];
        if (b == 0.0f)
        {
            throw std::domain_error("DivideOp::backward - Division by zero in gradient calculation at index " + std::to_string(i));
        }
        inputs[0]->grad[i] += output->grad[i] / b;
        inputs[1]->grad[i] -= (a * output->grad[i]) / (b * b);
    }
}

void ExpOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = std::exp(inputs[0]->data[i]);
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void ExpOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        inputs[0]->grad[i] += std::exp(inputs[0]->data[i]) * output->grad[i];
    }
}

void TanhOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = std::tanh(inputs[0]->data[i]);
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void TanhOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        float t = std::tanh(inputs[0]->data[i]);
        inputs[0]->grad[i] += (1.0f - t * t) * output->grad[i];
    }
}

void ReluOp::forward()
{
    check_one_input(inputs);
    output = std::make_shared<Tensor>(inputs[0]->shape);
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        output->data[i] = (inputs[0]->data[i] > 0.0f) ? inputs[0]->data[i] : 0.0f;
    }
    output->op = shared_from_this();
    output->children = inputs;
}

void ReluOp::backward()
{
    int sz = output->size();
    for (int i = 0; i < sz; i++)
    {
        inputs[0]->grad[i] += (inputs[0]->data[i] > 0.0f) ? output->grad[i] : 0.0f;
    }
}

void SumOp::forward()
{
    check_one_input(inputs);
    auto in = inputs[0];
    output = std::make_shared<Tensor>(std::vector<int>{1});
    float total = 0.0f;
    for (auto v : in->data)
        total += v;
    output->data[0] = total;
    output->op = shared_from_this();
    output->children = inputs;
}

void SumOp::backward()
{
    auto in = inputs[0];
    float grad_val = output->grad[0];
    for (auto &g : in->grad)
    {
        g += grad_val;
    }
}

void StackOp::forward()
{
    // Suppose each input is shape [1]. The output is shape [N], where N = inputs.size().
    int N = (int)inputs.size();
    output = std::make_shared<Tensor>(std::vector<int>{N});

    for (int i = 0; i < N; i++)
    {
        // Copy the single value from inputs[i] into output->data[i]
        // inputs[i]->data[0] should exist since it's shape [1]
        output->data[i] = inputs[i]->data[0];
    }

    output->op = shared_from_this();
    output->children = inputs;
}

void StackOp::backward()
{
    int N = (int)inputs.size();
    for (int i = 0; i < N; i++)
    {
        inputs[i]->grad[0] += output->grad[i];
    }
}
