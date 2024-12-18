// nn.cpp

#include "nn.h"
#include "tensor.h"

#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

float make_random()
{
    // Create a random float between -1 and 1
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    random = 2 * random - 1;
    return random;
}

Neuron::Neuron(int in_features, bool nonlin) : in_features(in_features), nonlin(nonlin)
{
    // Initialize the weights and bias with random values
    weights.resize(in_features);
    for (int i = 0; i < in_features; i++)
    {
        weights[i] = std::make_shared<Tensor>(make_random());
    }
    bias = std::make_shared<Tensor>(make_random());
}

std::vector<std::shared_ptr<Tensor>> Neuron::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    if (weights.size() != input.size())
    {
        throw std::invalid_argument("Input size does not match weights size.");
    }

    // Forward pass: w * x + b using Tensor operations
    // Start with bias
    std::shared_ptr<Tensor> sm = bias;

    // Accumulate w * x
    for (int i = 0; i < in_features; i++)
    {
        sm = sm + (input[i] * weights[i]);
    }

    // Apply activation if needed
    if (nonlin)
    {
        sm = sm->tanh();
    }

    sm->label = "Neuron";
    return {sm};
}

std::vector<std::shared_ptr<Tensor>> Neuron::parameters()
{
    std::vector<std::shared_ptr<Tensor>> params = weights;
    params.push_back(bias);
    return params;
}

Layer::Layer(int in_features, int out_features, bool nonlin) : in_features(in_features), out_features(out_features), nonlin(nonlin)
{
    // Initialize neurons for the layer
    neurons.resize(out_features);
    for (int i = 0; i < out_features; i++)
    {
        neurons[i] = std::make_shared<Neuron>(in_features, nonlin);
    }
}

std::vector<std::shared_ptr<Tensor>> Layer::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    std::vector<std::shared_ptr<Tensor>> outputs;
    outputs.reserve(out_features);
    for (auto &neuron : neurons)
    {
        auto neuron_output = (*neuron)(input);
        outputs.insert(outputs.end(), neuron_output.begin(), neuron_output.end());
    }
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> Layer::parameters()
{
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &neuron : neurons)
    {
        auto neuron_params = neuron->parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

MLP::MLP(int input_size, const std::vector<int> &layer_sizes)
{
    if (layer_sizes.empty())
    {
        throw std::invalid_argument("MLP must have at least one layer.");
    }

    int in_size = input_size;
    for (size_t i = 0; i < layer_sizes.size(); i++)
    {
        bool nonlin = (i != layer_sizes.size() - 1); // Nonlinear except last layer
        layers.push_back(std::make_shared<Layer>(in_size, layer_sizes[i], nonlin));
        in_size = layer_sizes[i];
    }
}

std::vector<std::shared_ptr<Tensor>> MLP::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    auto x = input;
    for (auto &layer : layers)
    {
        x = (*layer)(x);
    }
    return x;
}

std::vector<std::shared_ptr<Tensor>> MLP::parameters()
{
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &layer : layers)
    {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

std::ostream &operator<<(std::ostream &os, const Layer &layer)
{
    os << "Layer(" << layer.in_features << "->" << layer.out_features << ", nonlin=" << (layer.nonlin ? "True" : "False") << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const MLP &mlp)
{
    os << "MLP of [";
    for (size_t i = 0; i < mlp.layers.size(); i++)
    {
        os << *(mlp.layers[i]);
        if (i != mlp.layers.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}
