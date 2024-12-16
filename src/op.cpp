#include <math.h>
#include <stdexcept> // Include for standard exception types

#include "op.h"
#include "tensor.h" // Now Tensor is fully defined

// Define AddOp's forward method
VALUE_TYPE AddOp::forward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("AddOp::forward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    return inputs[0]->data + inputs[1]->data;
}

// Define AddOp's backward method
void AddOp::backward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("AddOp::backward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("AddOp::backward - Output tensor is null.");
    }

    inputs[0]->grad += output->grad;
    inputs[1]->grad += output->grad;
}

VALUE_TYPE SubtractOp::forward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("SubtractOp::forward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    return inputs[0]->data - inputs[1]->data;
}

void SubtractOp::backward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("SubtractOp::backward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("SubtractOp::backward - Output tensor is null.");
    }

    inputs[0]->grad += output->grad;
    inputs[1]->grad -= output->grad;
}

VALUE_TYPE MultiplyOp::forward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MultiplyOp::forward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    return inputs[0]->data * inputs[1]->data;
}

void MultiplyOp::backward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MultiplyOp::backward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("MultiplyOp::backward - Output tensor is null.");
    }

    inputs[0]->grad += inputs[1]->data * output->grad;
    inputs[1]->grad += inputs[0]->data * output->grad;
}

VALUE_TYPE DivideOp::forward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("DivideOp::forward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    // Additional check to prevent division by zero
    if (inputs[1]->data == 0.0f)
    {
        throw std::domain_error("DivideOp::forward - Division by zero.");
    }

    return inputs[0]->data / inputs[1]->data;
}

void DivideOp::backward()
{
    // Check that there are exactly two inputs
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("DivideOp::backward - Expected 2 inputs, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("DivideOp::backward - Output tensor is null.");
    }

    // Additional check to prevent division by zero in gradient calculation
    if (inputs[1]->data == 0.0f)
    {
        throw std::domain_error("DivideOp::backward - Division by zero in gradient calculation.");
    }

    inputs[0]->grad += output->grad / inputs[1]->data;
    inputs[1]->grad -= (inputs[0]->data * output->grad) / (inputs[1]->data * inputs[1]->data);
}

VALUE_TYPE ExpOp::forward()
{
    // Check that there is exactly one input
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ExpOp::forward - Expected 1 input, received " + std::to_string(inputs.size()) + ".");
    }

    return exp(inputs[0]->data);
}

void ExpOp::backward()
{
    // Check that there is exactly one input
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ExpOp::backward - Expected 1 input, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("ExpOp::backward - Output tensor is null.");
    }

    inputs[0]->grad += exp(inputs[0]->data) * output->grad;
}

VALUE_TYPE TanhOp::forward()
{
    // Check that there is exactly one input
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("TanhOp::forward - Expected 1 input, received " + std::to_string(inputs.size()) + ".");
    }

    return tanh(inputs[0]->data);
}

void TanhOp::backward()
{
    // Check that there is exactly one input
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("TanhOp::backward - Expected 1 input, received " + std::to_string(inputs.size()) + ".");
    }

    // Check that output is not null
    if (!output)
    {
        throw std::runtime_error("TanhOp::backward - Output tensor is null.");
    }

    inputs[0]->grad += (1.0f - tanh(inputs[0]->data) * tanh(inputs[0]->data)) * output->grad;
}
