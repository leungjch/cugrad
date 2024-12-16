#include "nn.h"
#include "tensor.h"

#include <iostream>
#include <memory>
#include <random>

float make_random()
{
    // Create a random float between -1 and 1
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    random = 2 * random - 1;
    return random;
}

Neuron::Neuron(int in_features, bool nonlin) : in_features(in_features), nonlin(nonlin)
{
    // Initialize the weights and bias
    // Random initialization
    weights.resize(in_features);
    for (int i = 0; i < in_features; i++)
    {
        // Create a random float between -1 and 1
        auto t = std::make_shared<Tensor>(make_random());
        weights[i] = t;
    }
    bias = std::make_shared<Tensor>(make_random());
}

// compute w * x + b
std::vector<std::shared_ptr<Tensor>> Neuron::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    // Check size of input == in_features
    if (weights.size() != input.size())
    { // put this into a string
        std::string error = "Input size of " + std::to_string(input.size()) + " does not match weights size of " + std::to_string(weights.size());
        throw std::invalid_argument(error);
    }

    // Implement the forward pass
    float sm = 0.0;
    for (int i = 0; i < in_features; i++)
    {
        sm += input[i]->data * weights[i]->data;
    }
    sm += bias->data;

    auto output = std::make_shared<Tensor>(sm);
    if (nonlin)
    {
        output = std::make_shared<Tensor>(sm)->tanh();
    }
    output->label = "Neuron";
    return {output};
}

std::vector<std::shared_ptr<Tensor>> Neuron::parameters()
{
    // Return the parameters ( weights and bias included )
    std::vector<std::shared_ptr<Tensor>> params;
    params.insert(params.end(), weights.begin(), weights.end());
    return params;
}

Layer::Layer(int in_features, int out_features) : in_features(in_features), out_features(out_features)
{
    // Initialize the weights and bias
    // Random initialization
    neurons.resize(out_features);
    for (int i = 0; i < out_features; i++)
    {
        auto n = std::make_shared<Neuron>(in_features);
        neurons[i] = n;
    }
}

std::vector<std::shared_ptr<Tensor>> Layer::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    outputs.reserve(out_features);
    for (int i = 0; i < neurons.size(); i++)
    {
        auto output = neurons[i]->operator()(input);
        outputs.insert(outputs.end(), output.begin(), output.end());
    }

    return outputs;
}

std::vector<std::shared_ptr<Tensor>> Layer::parameters()
{
    // Return the parameters ( weights and bias included )
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &neuron : neurons)
    {
        auto n_params = neuron->parameters();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

MLP::MLP(int in_features, int out_features)
{
}

std::vector<std::shared_ptr<Tensor>> MLP::operator()(std::vector<std::shared_ptr<Tensor>> input)
{
    auto x = input;
    for (auto &layer : layers)
    {
        x = layer->operator()(x);
    }
    return x;
}

std::vector<std::shared_ptr<Tensor>> MLP::parameters()
{
    // Return the parameters ( weights and bias included )
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &layer : layers)
    {
        auto l_params = layer->parameters();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}
