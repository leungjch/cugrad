#ifndef OP_H
#define OP_H

#include <iostream>
#include <memory>
#include <vector>
#include <cassert>

#include "value.h"
// Remove the following line to prevent circular dependency
// #include "tensor.h"

class Tensor; // Forward declaration

class Op
{
public:
    Op(const std::vector<std::shared_ptr<Tensor>> &inputs) : inputs(inputs) {}

    virtual VALUE_TYPE forward() = 0;
    virtual void backward() = 0;

    virtual ~Op() {}

    std::shared_ptr<Tensor> output;
    std::vector<std::shared_ptr<Tensor>> inputs;
};

class AddOp : public Op
{
public:
    // Constructor for AddOp
    AddOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

#endif // OP_H
