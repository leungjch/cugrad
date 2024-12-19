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

class Op : public std::enable_shared_from_this<Op>
{
public:
    virtual void forward() = 0;
    virtual void backward() = 0;

    Op(const std::vector<std::shared_ptr<Tensor>> &inputs, std::string op_type = "") : inputs(inputs), op_type(op_type)
    {
    }

    virtual ~Op() {}

    std::shared_ptr<Tensor> output;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::string op_type;
};

class AddOp : public Op
{
public:
    // Constructor for AddOp
    AddOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "add") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class SubtractOp : public Op
{
public:
    // Constructor for SubtractOp
    SubtractOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "sub") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class MultiplyOp : public Op
{
public:
    // Constructor for MultiplyOp
    MultiplyOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "mul") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class DivideOp : public Op
{
public:
    // Constructor for DivideOp
    DivideOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "div") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class ExpOp : public Op
{
public:
    // Constructor for ExpOp
    ExpOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "exp") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class TanhOp : public Op
{
public:
    // Constructor for TanhOp
    TanhOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "tanh") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class ReluOp : public Op
{
public:
    // Constructor for ReluOp
    ReluOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "relu") {}

    // Declare the methods
    void forward() override;
    void backward() override;
};

class SumOp : public Op
{
public:
    SumOp(const std::vector<std::shared_ptr<Tensor>> &inputs) : Op(inputs, "sum") {}
    void forward() override;
    void backward() override;
};

class StackOp : public Op
{
public:
    // Constructor: pass a list of single-value tensors.
    StackOp(const std::vector<std::shared_ptr<Tensor>> &inputs)
        : Op(inputs, "stack") {}

    void forward() override;
    void backward() override;
};

#endif // OP_H
