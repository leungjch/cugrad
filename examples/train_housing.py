import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from cugrad.tensor import Tensor
from cugrad.nn import MLP
from cugrad.optimizer import SGD
from cugrad import set_device, DeviceType

set_device(DeviceType.CPU)

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = data.data
y = data.target

# Normalize features
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-7
X_norm = (X - X_mean) / X_std

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Convert to lists of Tensors for easy iteration
train_data = [(Tensor(x), Tensor([y_])) for x, y_ in zip(X_train, y_train)]
test_data = [(Tensor(x), Tensor([y_])) for x, y_ in zip(X_test, y_test)]

# Define a small MLP: 13 input features -> [16, 16] hidden layers -> 1 output
model = MLP(input_size=8, layer_sizes=[16, 16, 1])



# Mean Squared Error (MSE) loss
def mse_loss(predictions, targets):
    # predictions and targets are lists of Tensors
    losses = []
    for pred, target in zip(predictions, targets):
        error = pred - target
        losses.append(error * error)
    out = losses[0]
    for i in range(1, len(losses)):
        out = out + losses[i]
    out = out * Tensor([1.0 / len(losses)])
    return out

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Shuffle data
    np.random.shuffle(train_data)
    inputs = [d[0] for d in train_data]
    targets = [d[1] for d in train_data]

    model.zero_grad()
    
    # Forward pass
    predictions = [model(inp) for inp in inputs]
    loss = mse_loss(predictions, targets)

    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data[0]:.4f}")

# Evaluate on the test set
test_inps = [d[0] for d in test_data]
test_tgts = [d[1] for d in test_data]
test_preds = [model(inp) for inp in test_inps]
test_loss = mse_loss(test_preds, test_tgts)
print(f"\nTest Loss: {test_loss.data[0]:.4f}")

# Print a few predictions vs targets
for i in range(5):
    print(f"Input: {test_inps[i].data}, Predicted: {test_preds[i].data[0]:.4f}, Target: {test_tgts[i].data[0]:.4f}")
