#include "tensor.h"
#include "op.h"

#include <memory>
#include <stack>
#include <unordered_set>
#include <algorithm>

#include "tensor.h"
#include "op.h"

std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
{
    os << "Tensor(" << tensor.data << ", grad=" << tensor.grad << ", label=" << tensor.label << ")";
    return os;
}

std::shared_ptr<Tensor> make_tensor(std::shared_ptr<Op> op, std::vector<std::shared_ptr<Tensor>> inputs)
{
    // Perform the forward operation
    VALUE_TYPE result = op->forward();

    // Create the output tensor
    auto result_tensor = std::make_shared<Tensor>(result, op, inputs);
    op->output = result_tensor;
    return result_tensor;
}

std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor> &other)
{
    auto add_op = std::make_shared<AddOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    return make_tensor(add_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor> &other)
{
    auto sub_op = std::make_shared<SubtractOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    return make_tensor(sub_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor> &other)
{
    auto mul_op = std::make_shared<MultiplyOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    return make_tensor(mul_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::operator/(const std::shared_ptr<Tensor> &other)
{
    auto div_op = std::make_shared<DivideOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    return make_tensor(div_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::tanh()
{
    auto tanh_op = std::make_shared<TanhOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    return make_tensor(tanh_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::exp()
{
    auto exp_op = std::make_shared<ExpOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
    return make_tensor(exp_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this()});
}

void Tensor::backward()
{
    // Initialize the gradient of the output tensor to 1.0
    grad = 1.0;

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
    grad = 0.0;

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
