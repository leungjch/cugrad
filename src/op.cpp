#include "op.h"
#include "tensor.h" // Now Tensor is fully defined

// Define AddOp's forward method
VALUE_TYPE AddOp::forward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);

    return inputs[0]->data + inputs[1]->data;
}

// Define AddOp's backward method
void AddOp::backward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);
    assert(output);

    inputs[0]->grad += output->grad;
    inputs[1]->grad += output->grad;
}

VALUE_TYPE SubtractOp::forward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);

    return inputs[0]->data - inputs[1]->data;
}

void SubtractOp::backward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);
    assert(output);

    inputs[0]->grad += output->grad;
    inputs[1]->grad -= output->grad;
}

VALUE_TYPE MultiplyOp::forward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);

    return inputs[0]->data * inputs[1]->data;
}

void MultiplyOp::backward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);
    assert(output);

    inputs[0]->grad += inputs[1]->data * output->grad;
    inputs[1]->grad += inputs[0]->data * output->grad;
}

VALUE_TYPE DivideOp::forward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);

    return inputs[0]->data / inputs[1]->data;
}

void DivideOp::backward()
{
    // Assert that there are exactly two inputs
    assert(inputs.size() == 2);
    assert(output);

    inputs[0]->grad += output->grad / inputs[1]->data;
    inputs[1]->grad -= inputs[0]->data * output->grad / (inputs[1]->data * inputs[1]->data);
}
