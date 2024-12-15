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

    inputs[0]->grad += output->grad;
    inputs[1]->grad += output->grad;
}
