# train_xor.py

import unittest
from cugrad.tensor import Tensor
from cugrad.nn import MLP
import math

# Define the XOR inputs and corresponding targets
xor_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

xor_targets = [0.0, 1.0, 1.0, 0.0]

# Initialize the MLP with 2 inputs, one hidden layer with 3 neurons, and 1 output
model = MLP(input_size=2, layer_sizes=[3, 1])

# Define the Mean Squared Error (MSE) loss function
def mse_loss(predictions, targets):
    loss = Tensor(0.0)
    for pred, target in zip(predictions, targets):
        error = pred - Tensor(target)
        loss = loss + (error * error)
    return loss

# Set training parameters
learning_rate = 0.005
epochs = 1000

for epoch in range(epochs):
    epoch_loss = Tensor(0.0)
    model.zero_grad()  # Reset gradients before each epoch
    
    # Forward pass
    predictions = []
    for input_data in xor_inputs:
        input_tensors = [Tensor(x) for x in input_data]
        output = model(input_tensors)[0]  # Assuming single output
        print("data", output.data)
        predictions.append(output)
    
    # Compute loss
    loss = mse_loss(predictions, xor_targets)
    epoch_loss = loss.data
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    for param in model.parameters():
        param.data -= learning_rate * param.grad
    
    # Optionally, print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("\nTrained Model Predictions:")
for input_data, target in zip(xor_inputs, xor_targets):
    input_tensors = [Tensor(x) for x in input_data]
    output = model(input_tensors)[0]
    print(f"Input: {input_data}, Predicted: {output.data:.4f}, Target: {target}")
