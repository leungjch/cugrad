#ifndef TENSOR_H
#define TENSOR_H

#include "value.h"
#include "op.h" // Include after forward declarations

#include <iostream>
#include <vector>
#include <memory>

class Op;

class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    float data;
    float grad;
    std::string label;                             // Label for debugging
    std::shared_ptr<Op> op;                        // Operation that created this Tensor
    std::vector<std::shared_ptr<Tensor>> children; // Children tensors

    // Constructors
    Tensor(VALUE_TYPE data, std::shared_ptr<Op> op = nullptr, std::vector<std::shared_ptr<Tensor>> children = {})
        : data(data), grad(0.0f), op(op), children(children) {}

    Tensor() : data(0.0f), grad(0.0f), op(nullptr), children({}) {}

    // Operator Overloads
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &other);

    // Backward Pass
    void backward();

    // Topological Sort Utility
    void topological_sort(std::vector<std::shared_ptr<Tensor>> &ordering);

    // Reset gradients (useful for multiple backward passes)
    void zero_grad();

    // Friend function for ostream
    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
};

#endif // TENSOR_H
