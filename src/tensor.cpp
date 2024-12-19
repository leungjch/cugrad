// tensor.cpp

#include "tensor.h"
#include "op.h"

#include <memory>
#include <stack>
#include <unordered_set>
#include <algorithm>

#include "tensor.h"
#include "op.h"

// Constructs a tensor of given shape with optional initialization
Tensor::Tensor(const std::vector<int> &shape, float init_val,
               std::shared_ptr<Op> op,
               std::vector<std::shared_ptr<Tensor>> children,
               DeviceType device)
    : shape(shape), op(op), children(children), device(device)
{
    int total_size = 1;
    for (auto s : shape)
    {
        if (s <= 0)
        {
            throw std::invalid_argument("All dimensions must be positive.");
        }
        total_size *= s;
    }
    data.resize(total_size, init_val);
    grad.resize(total_size, 0.0f);
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
{
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape.size(); i++)
    {
        os << tensor.shape[i];
        if (i < tensor.shape.size() - 1)
            os << ", ";
    }
    os << "], data=[";
    int sz = tensor.size();
    for (int i = 0; i < sz; i++)
    {
        os << tensor.data[i];
        if (i < sz - 1)
            os << ", ";
    }
    os << "], grad=[";
    for (int i = 0; i < sz; i++)
    {
        os << tensor.grad[i];
        if (i < sz - 1)
            os << ", ";
    }
    os << "])";
    return os;
}

int Tensor::size() const
{
    int sz = 1;
    for (int s : shape)
    {
        sz *= s;
    }
    return sz;
}

std::shared_ptr<Tensor> Tensor::scalar_tensor(float val)
{
    auto t = std::make_shared<Tensor>(std::vector<int>{1}, val);
    return t;
}

// Check that two tensors have the same shape
void Tensor::check_same_shape(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
    if (a->shape != b->shape)
    {
        throw std::invalid_argument("Shapes must match for this operation.");
    }
}

// Implement element-wise ops
std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor> &other)
{
    check_same_shape(shared_from_this(), other);
    // Create AddOp etc. Here assume we have AddOp adapted for arrays
    std::shared_ptr<Op> add_op = std::make_shared<AddOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    // AddOp forward will fill the output->data
    add_op->forward();
    return add_op->output;
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor> &other)
{
    check_same_shape(shared_from_this(), other);
    std::shared_ptr<Op> sub_op = std::make_shared<SubtractOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    sub_op->forward();
    return sub_op->output;
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor> &other)
{
    check_same_shape(shared_from_this(), other);
    std::shared_ptr<Op> mul_op = std::make_shared<MultiplyOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    mul_op->forward();
    return mul_op->output;
}

std::shared_ptr<Tensor> Tensor::operator/(const std::shared_ptr<Tensor> &other)
{
    check_same_shape(shared_from_this(), other);
    std::shared_ptr<Op> div_op = std::make_shared<DivideOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    div_op->forward();
    return div_op->output;
}

// Scalar operations: create a scalar tensor with the same shape and then do element-wise op
std::shared_ptr<Tensor> Tensor::operator+(float scalar)
{
    auto s = scalar_tensor(scalar);
    return (*this) + s;
}

std::shared_ptr<Tensor> Tensor::operator-(float scalar)
{
    auto s = scalar_tensor(scalar);
    return (*this) - s;
}

std::shared_ptr<Tensor> Tensor::operator*(float scalar)
{
    auto s = scalar_tensor(scalar);
    return (*this) * s;
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar)
{
    auto s = scalar_tensor(scalar);
    return (*this) / s;
}

std::shared_ptr<Tensor> Tensor::tanh()
{
    std::shared_ptr<Op> tanh_op = std::make_shared<TanhOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    tanh_op->forward();
    return tanh_op->output;
}

std::shared_ptr<Tensor> Tensor::relu()
{
    std::shared_ptr<Op> relu_op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    relu_op->forward();
    return relu_op->output;
}

std::shared_ptr<Tensor> Tensor::exp()
{
    std::shared_ptr<Op> exp_op = std::make_shared<ExpOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    exp_op->forward();
    return exp_op->output;
}

std::shared_ptr<Tensor> Tensor::sum()
{
    auto op_ = std::make_shared<SumOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    op_->forward();
    return op_->output;
}

void Tensor::backward()
{
    // Initialize the gradient of the output tensor to 1.0
    std::fill(grad.begin(), grad.end(), 1.0);

    // Get the topological ordering of the compute graph
    std::vector<std::shared_ptr<Tensor>> ordering;
    topological_sort(ordering);

    for (auto it = ordering.rbegin(); it != ordering.rend(); ++it)
    {
        auto tensor = *it;
        // If the tensor has an operation
        if (tensor->op)
        {
            // Perform the backward pass
            tensor->op->backward();
        }
    }
}

void Tensor::zero_grad()
{
    std::fill(grad.begin(), grad.end(), 0.0);

    // Recursively zero the gradients of the children
    for (auto child : children)
    {
        child->zero_grad();
    }
}

// Builds a topological ordering of the compute graph
void Tensor::topological_sort(std::vector<std::shared_ptr<Tensor>> &ordering)
{
    // Visited set
    std::unordered_set<std::shared_ptr<Tensor>> visited;

    // Stack
    std::stack<std::shared_ptr<Tensor>> stack;
    stack.push(shared_from_this());

    while (!stack.empty())
    {
        // Get the top element
        auto current = stack.top();

        // If the current node is not visited
        if (visited.find(current) == visited.end())
        {
            // Mark the current node as visited
            visited.insert(current);

            // Add children to the stack
            for (auto child : current->children)
            {
                if (visited.find(child) == visited.end())
                {
                    stack.push(child);
                }
            }
        }
        else
        {
            stack.pop();
            ordering.push_back(current);
        }
    }
}

// Global operator overloads implementation

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
    return a->operator+(b);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
    return a->operator-(b);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
    return a->operator*(b);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
    return a->operator/(b);
}
