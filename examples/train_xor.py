# train_xor.py

import unittest
from cugrad.tensor import Tensor
from cugrad import get_device, set_device, DeviceType
from cugrad.nn import MLP
from cugrad.optimizer import SGD
import math

# Set device
set_device(DeviceType.CUDA)

print("Get device: ", get_device())

# Define the XOR inputs and corresponding targets
xor_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

xor_targets = [0.0, 1.0, 1.0, 0.0]

# Initialize the MLP with 2 inputs, one hidden layer with 3 neurons, and 1 output
model = MLP(input_size=2, layer_sizes=[2, 1])

# Define the Mean Squared Error (MSE) loss function
def mse_loss(predictions, targets):
    loss = Tensor([0.0])
    for pred, target in zip(predictions, targets):
        target_tensor = Tensor([target])
        error = pred - target_tensor
        loss = loss + (error * error)
    return loss

# Set training parameters
learning_rate = 0.005
epochs = 1000

optimizer = SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    epoch_loss = Tensor([0.0])
    model.zero_grad()  # Reset gradients before each epoch
    
    # Forward pass
    predictions = []
    for input_data in xor_inputs:
        input_tensor = Tensor(input_data)

        output = model(input_tensor)
        predictions.append(output)
    
    # Compute loss
    loss = mse_loss(predictions, xor_targets)
    
    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        loss.to_device(DeviceType.CPU)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data} {loss.grad}")
        loss.to_device(DeviceType.CUDA)

print("\nTrained Model Predictions:")
for input_data, target in zip(xor_inputs, xor_targets):
    input_tensor = Tensor(input_data)
    output = model(input_tensor)
    output.to_device(DeviceType.CPU)
    print(f"Input: {input_data}, Predicted: {output.data[0]:.4f}, Target: {target}")
