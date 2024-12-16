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
    virtual VALUE_TYPE forward() = 0;
    virtual void backward() = 0;

    Op(const std::vector<std::shared_ptr<Tensor>> &inputs) : inputs(inputs)
    {
    }

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

class SubtractOp : public Op
{
public:
    // Constructor for SubtractOp
    SubtractOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

class MultiplyOp : public Op
{
public:
    // Constructor for MultiplyOp
    MultiplyOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

class DivideOp : public Op
{
public:
    // Constructor for DivideOp
    DivideOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

class ExpOp : public Op
{
public:
    // Constructor for ExpOp
    ExpOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

class TanhOp : public Op
{
public:
    // Constructor for TanhOp
    TanhOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs) {}

    // Declare the methods
    VALUE_TYPE forward() override;
    void backward() override;
};

#endif // OP_H
