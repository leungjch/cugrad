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
    weights = std::make_shared<Tensor>(std::vector<int>{in_features});
    // Fill weights->data with random values
    for (int i = 0; i < in_features; i++)
    {
        weights->data[i] = make_random();
    }

    bias = std::make_shared<Tensor>(std::vector<int>{1});
    bias->data[0] = make_random();
}

std::shared_ptr<Tensor> Neuron::operator()(std::shared_ptr<Tensor> input)
{
    // Input must have shape [in_features]
    if (input->shape.size() != 1 || input->shape[0] != in_features)
    {
        throw std::invalid_argument("Input to Neuron must have shape " + std::to_string(in_features) + " but got " + std::to_string(input->shape[0]));
    }

    // Output = bias + sum(input * weights)
    auto prod = input * weights; // shape [in_features]
    auto summed = prod->sum();   // shape [1]
    auto out = summed + bias;    // shape [1]

    if (nonlin)
    {
        out = out->tanh();
    }

    return out; // A single Tensor of shape [1]
}

std::vector<std::shared_ptr<Tensor>> Neuron::parameters()
{
    return {weights, bias};
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

std::shared_ptr<Tensor> Layer::operator()(std::shared_ptr<Tensor> input)
{
    std::vector<std::shared_ptr<Tensor>> neuron_outputs;
    neuron_outputs.reserve(out_features);

    for (int i = 0; i < out_features; i++)
    {
        // Each neuron returns a [1]-shaped tensor.
        auto neuron_output = (*neurons[i])(input);
        neuron_outputs.push_back(neuron_output);
    }

    // Use StackOp to combine these into a single [out_features]-shaped tensor
    auto stack_op = std::make_shared<StackOp>(neuron_outputs);
    stack_op->forward();
    // stack_op->forward() sets stack_op->output

    return stack_op->output; // This output now has a proper op and children set
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

std::shared_ptr<Tensor> MLP::operator()(std::shared_ptr<Tensor> input)
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
