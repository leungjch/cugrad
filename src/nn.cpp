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

Linear::Linear(int in_features, int out_features)
    : in_features(in_features), out_features(out_features)
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
    std::cout << "Made bias with random value" << bias->data << std::endl;
}

// compute w * x + b
std::shared_ptr<Tensor> Linear::forward(std::vector<std::shared_ptr<Tensor>> input)
{
    // Check size of input == in_features
    if (weights.size() != input.size())
    { // put this into a string
        std::string error = "Input size of " + std::to_string(input.size()) + " does not match weights size of " + std::to_string(weights.size());
        throw std::invalid_argument(error);
    }

    std::cout << "Linear::forward called with " << input.size() << " inputs." << std::endl;
    std::cout << "Weights size is " << weights.size() << std::endl;
    // Implement the forward pass
    float sm = 0.0;
    for (int i = 0; i < in_features; i++)
    {
        std::cout << "Input[" << i << "]: " << input[i]->data
                  << " * Weight[" << i << "]: " << weights[i]->data << std::endl;
        sm += input[i]->data * weights[i]->data;
    }
    std::cout << "Sum after multiplication: " << sm << std::endl;
    sm += bias->data;
    std::cout << "Adding bias: " << bias->data << ", Total: " << sm << std::endl;

    auto output = std::make_shared<Tensor>(sm);
    output->label = "Linear";
    return output;
}

void Linear::zero_grad()
{
    // Zero the gradients
    for (auto &weight : weights)
    {
        weight->zero_grad();
    }
    bias->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters()
{
    // Return the parameters ( weights and bias included )
    // I want to avoid copying the vectors, so I will return a reference
    std::vector<std::shared_ptr<Tensor>> params;
    params.insert(params.end(), weights.begin(), weights.end());
    return params;
}
