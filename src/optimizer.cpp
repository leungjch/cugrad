// optimizer.cpp

#include "optimizer.h"
#include "op_cuda.h"

#include <cstddef> // for size_t

#include "Tracy.hpp" // Include Tracy's header

// Optimizer Methods
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &parameters)
    : parameters(parameters)
{
}

Optimizer::~Optimizer() {}

void Optimizer::zero_grad()
{
    ZoneScopedN("Zero gradients"); // Named profiling zone
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
    ZoneScopedN("SGD Step"); // Named profiling zone
    for (auto &param : parameters)
    {
        int sz = param->size();
        if (param->device == DeviceType::CUDA)
        {
            // Ensure memory is allocated and data is on the device
            param->allocate_memory_on_device();

            // Launch CUDA kernel for SGD step
            sgd_step_cuda(param->d_data, param->d_grad, lr, sz);

            // Optionally, copy updated data back to host if needed
            // param->copy_to_host();
        }
        else // CPU
        {
            for (int i = 0; i < sz; i++)
            {
                param->data[i] -= lr * param->grad[i];
            }
        }
    }
}
