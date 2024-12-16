#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>

#include "tensor.h"
#include "op.h"

class Module
{
public:
    virtual std::vector<std::shared_ptr<Tensor>> operator()(std::vector<std::shared_ptr<Tensor>> input) = 0;
    virtual void zero_grad()
    {
        for (auto &param : parameters())
        {
            param->zero_grad();
        }
    }

    // Pure virtual method to retrieve parameters
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;

    virtual ~Module() {}
};

class Neuron : public Module
{
public:
    Neuron(int in_features, bool nonlin = true);

    std::vector<std::shared_ptr<Tensor>> operator()(std::vector<std::shared_ptr<Tensor>> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::vector<std::shared_ptr<Tensor>> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Op> activation;
    int in_features;
    bool nonlin;
};

class Layer : public Module
{
public:
    Layer(int in_features, int out_features);

    std::vector<std::shared_ptr<Tensor>> operator()(std::vector<std::shared_ptr<Tensor>> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::vector<std::shared_ptr<Tensor>> weights;
    std::shared_ptr<Tensor> bias;
    int in_features;
    int out_features;
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Neuron>> neurons;
};

class MLP : public Module
{
public:
    MLP(int in_features, int out_features);

    std::vector<std::shared_ptr<Tensor>> operator()(std::vector<std::shared_ptr<Tensor>> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Layer> hidden;
    std::shared_ptr<Layer> output;

    std::vector<std::shared_ptr<Layer>> layers;
};

#endif // NN_H
