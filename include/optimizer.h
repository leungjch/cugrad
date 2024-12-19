#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include <functional> // for std::reference_wrapper
#include "tensor.h"

class Optimizer
{
public:
    Optimizer(const std::vector<std::shared_ptr<Tensor>> &parameters);
    virtual ~Optimizer();

    virtual void step() = 0;
    virtual void zero_grad();

protected:
    std::vector<std::shared_ptr<Tensor>> parameters;
};

class SGD : public Optimizer
{
public:
    SGD(const std::vector<std::shared_ptr<Tensor>> &parameters, float lr);
    void step() override;

private:
    float lr;
};

#endif // OPTIMIZER_H
