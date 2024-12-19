// nn.h

#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>
#include <memory>
#include "tensor.h"
#include "op.h"

class Module
{
public:
    virtual void zero_grad()
    {
        for (auto &param : parameters())
        {
            param->zero_grad();
        }
    }

    // Pure virtual method to retrieve parameters
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
    // Forward pass
    virtual std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) = 0;

    virtual ~Module() {}
};

class Neuron : public Module
{
public:
    Neuron(int in_features, bool nonlin = true);

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Op> activation;
    int in_features;
    bool nonlin;
};

class Layer : public Module
{
public:
    Layer(int in_features, int out_features, bool nonlin = true); // Added nonlin parameter
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    int in_features;
    int out_features;
    bool nonlin;
    std::vector<std::shared_ptr<Neuron>> neurons;
};

class MLP : public Module
{
public:
    MLP(int input_size, const std::vector<int> &layer_sizes);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::vector<std::shared_ptr<Layer>> layers;
};

#endif // NN_H
