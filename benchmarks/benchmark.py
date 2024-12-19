import time
import torch
import micrograd.engine as mg     # If micrograd is installed
from cugrad.nn import MLP as CuMLP
from cugrad.tensor import Tensor as CuTensor
from cugrad.optimizer import SGD as CuSGD
from cugrad import DeviceType
from device_manager import DeviceManager

# PyTorch model
import torch.nn as nn
import torch.optim as optim

def benchmark_cugrad(device_type='CPU'):
    # Set device
    if device_type == 'CUDA':
        DeviceManager.get_instance().set_current_device(DeviceType.CUDA)
    else:
        DeviceManager.get_instance().set_current_device(DeviceType.CPU)
    
    # Create MLP in cugrad
    model = CuMLP(input_size=1000, layer_sizes=[512, 512, 512, 1])
    optimizer = CuSGD(model.parameters(), lr=0.01)

    # Create random input and target
    input_data = [0.0]*1000
    # Just random initialize - more sophisticated might use random module
    import random
    for i in range(1000):
        input_data[i] = random.random()
    input_tensor = CuTensor([1000])
    input_tensor.data = input_data
    target = CuTensor([1])
    target.data = [0.5]  # Arbitrary target
    
    # Warm up
    for _ in range(10):
        out = model(input_tensor)
        loss = (out - target)*(out - target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Timing
    start = time.time()
    for _ in range(100):
        out = model(input_tensor)
        loss = (out - target)*(out - target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print(f"cugrad {device_type} time: {end - start:.4f}s")

def benchmark_pytorch(use_cuda=False):
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    # PyTorch model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.Tanh(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 1)
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Random input
    input_data = torch.rand(1000, device=device)
    target = torch.tensor([0.5], device=device)

    # Warm up
    for _ in range(10):
        out = model(input_data)
        loss = (out - target)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    start = time.time()
    for _ in range(100):
        out = model(input_data)
        loss = (out - target)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print(f"PyTorch {'CUDA' if use_cuda else 'CPU'} time: {end - start:.4f}s")

def benchmark_micrograd():
    # micrograd MLP
    # micrograd uses a Value class not Tensors, define a similar MLP:
    class MicrogradMLP:
        def __init__(self):
            import math
            # Using small weights for simplicity
            self.W1 = [[mg.Value(random.random()) for _ in range(1000)] for _ in range(512)]
            self.b1 = [mg.Value(random.random()) for _ in range(512)]
            self.W2 = [[mg.Value(random.random()) for _ in range(512)] for _ in range(512)]
            self.b2 = [mg.Value(random.random()) for _ in range(512)]
            self.W3 = [[mg.Value(random.random()) for _ in range(512)] for _ in range(512)]
            self.b3 = [mg.Value(random.random()) for _ in range(512)]
            self.W4 = [[mg.Value(random.random()) for _ in range(512)] for _ in range(1)]
            self.b4 = [mg.Value(random.random())]

        def forward(self, x):
            # x is a list of floats -> convert to mg.Value
            xv = [mg.Value(v) for v in x]
            def linear(x, W, b):
                out = []
                for row, bb in zip(W, b):
                    sum_ = mg.Value(0)
                    for xi, wi in zip(x, row):
                        sum_ = sum_ + xi*wi
                    sum_ = sum_ + bb
                    out.append(sum_)
                return out

            def tanh_layer(h):
                return [o.tanh() for o in h]

            h1 = tanh_layer(linear(xv, self.W1, self.b1))
            h2 = tanh_layer(linear(h1, self.W2, self.b2))
            h3 = tanh_layer(linear(h2, self.W3, self.b3))
            h4 = linear(h3, self.W4, self.b4)
            return h4[0]

        def parameters(self):
            # Flatten all parameters
            return [p for row in self.W1 for p in row]+self.b1 + \
                   [p for row in self.W2 for p in row]+self.b2 + \
                   [p for row in self.W3 for p in row]+self.b3 + \
                   [p for row in self.W4 for p in row]+self.b4

    model = MicrogradMLP()
    # Random input
    x = [random.random() for _ in range(1000)]
    target = mg.Value(0.5)

    # Warm up
    for _ in range(10):
        out = model.forward(x)
        loss = (out - target)*(out - target)
        # zero_grad
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        # update
        for p in model.parameters():
            p.data -= 0.01*p.grad

    start = time.time()
    for _ in range(100):
        out = model.forward(x)
        loss = (out - target)*(out - target)
        # zero_grad
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        # update
        for p in model.parameters():
            p.data -= 0.01*p.grad
    end = time.time()
    print(f"micrograd CPU time: {end - start:.4f}s")

if __name__ == "__main__":
    # Benchmark CPU cugrad
    benchmark_cugrad(device_type='CPU')

    # If you have implemented CUDA kernels and GPU support
    benchmark_cugrad(device_type='CUDA')

    # Benchmark PyTorch CPU
    benchmark_pytorch(use_cuda=False)

    # Benchmark PyTorch CUDA (if available)
    if torch.cuda.is_available():
        benchmark_pytorch(use_cuda=True)

    # Benchmark micrograd CPU
    benchmark_micrograd()
