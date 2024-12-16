#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>

#include "tensor.h"
#include "op.h"

class Module
{
public:
    virtual std::shared_ptr<Tensor> forward(std::vector<std::shared_ptr<Tensor>> input) = 0;
    virtual void zero_grad() = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;

    virtual ~Module() {}
};

class Linear : public Module
{
public:
    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::vector<std::shared_ptr<Tensor>> input) override;
    void zero_grad() override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::vector<std::shared_ptr<Tensor>> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Op> activation;
    int in_features;
    int out_features;
};

#endif // NN_H
