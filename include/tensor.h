#ifndef TENSOR_H
#define TENSOR_H

#include "value.h"
#include "op.h"
#include "device.h"

#include <iostream>
#include <vector>
#include <memory>

class Op;

class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> grad;

    DeviceType device;

    std::string label;                             // Label for debugging
    std::shared_ptr<Op> op;                        // Operation that created this Tensor
    std::vector<std::shared_ptr<Tensor>> children; // Children tensors

    Tensor() : device(DeviceType::CPU), op(nullptr)
    {
        shape = {1};
        data.resize(1, 0.0f);
        grad.resize(1, 0.0f);
    }

    // Constructors
    Tensor(const std::vector<int> &shape, float init_val = 0.0f,
           std::shared_ptr<Op> op = nullptr,
           std::vector<std::shared_ptr<Tensor>> children = {},
           DeviceType device = DeviceType::CPU);

    // Operator Overloads
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &other);

    std::shared_ptr<Tensor> operator+(float scalar);
    std::shared_ptr<Tensor> operator*(float scalar);
    std::shared_ptr<Tensor> operator-(float scalar);
    std::shared_ptr<Tensor> operator/(float scalar);

    std::shared_ptr<Tensor> tanh();
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> exp();

    std::shared_ptr<Tensor> sum();

    // Backward Pass
    void backward();

    // Topological Sort Utility
    void topological_sort(std::vector<std::shared_ptr<Tensor>> &ordering);

    // Reset gradients (useful for multiple backward passes)
    void zero_grad();

    // Friend function for ostream
    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    static void check_same_shape(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);

    // Helper function to define a scalar
    static std::shared_ptr<Tensor> scalar_tensor(float val);

    int size() const;
};

// Global operator overloads for std::shared_ptr<Tensor>
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);

#endif // TENSOR_H
