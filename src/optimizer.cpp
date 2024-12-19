#include "optimizer.h"
#include <cstddef> // for size_t

// Optimizer Methods
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &parameters)
    : parameters(parameters)
{
}

Optimizer::~Optimizer() {}

void Optimizer::zero_grad()
{
    for (auto &param : parameters)
    {
        param->zero_grad();
    }
}

// SGD Methods
SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &parameters, float lr)
    : Optimizer(parameters), lr(lr)
{
}

void SGD::step()
{
    for (auto &param : parameters)
    {
        int sz = param->size();
        for (int i = 0; i < sz; i++)
        {
            param->data[i] -= lr * param->grad[i];
        }
    }
}
