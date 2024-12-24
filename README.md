# cugrad

[![Build and Test](https://github.com/leungjch/cugrad/actions/workflows/workflow.yml/badge.svg)](https://github.com/leungjch/cugrad/actions/workflows/workflow.yml)

![](examples/graphs/layer_compute_graph.png)

cugrad is a simple automatic differentiation library written in C++ with Python bindings resembling the PyTorch API. It supports CPU and CUDA backends.

The engine is a C++ extension of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), adding multidimensional tensor support and CUDA. This is mainly a learning project to understand how automatic differentiation and GPU acceleration works from scratch and not intended for production use.

## Building

To build the cugrad Python bindings, run
```bash.
pip install .
```

To run tests, run
```bash.
pip install pytest
pytest tests/*
```

## Usage

Python bindings provide a very similar API to Pytorch. Here is an example MLP that learns XOR:

https://github.com/leungjch/cugrad/blob/c9096292901ced4a748e12297dfc1dd78e5c50ca/examples/train_xor.py#L1-L72

Output:
```
Epoch 100/1000, Loss: [0.9880131483078003]
Epoch 200/1000, Loss: [0.9438388347625732]
Epoch 300/1000, Loss: [0.8765984773635864]
Epoch 400/1000, Loss: [0.7637998461723328]
Epoch 500/1000, Loss: [0.5813338160514832]
Epoch 600/1000, Loss: [0.3450694680213928]
Epoch 700/1000, Loss: [0.1463480442762375]
Epoch 800/1000, Loss: [0.04524468630552292]
Epoch 900/1000, Loss: [0.011270132847130299]
Epoch 1000/1000, Loss: [0.002491143997758627]

Trained Model Predictions:
Input: [0.0, 0.0], Predicted: 0.0209, Target: 0.0
Input: [0.0, 1.0], Predicted: 0.9858, Target: 1.0
Input: [1.0, 0.0], Predicted: 0.9683, Target: 1.0
Input: [1.0, 1.0], Predicted: 0.0285, Target: 0.0
```

See also [demo.ipynb](https://github.com/leungjch/cugrad/blob/main/examples/demo.ipynb) adapted from [micrograd](https://github.com/karpathy/micrograd/blob/master/demo.ipynb) which trains a 2D classifier:

![image](https://github.com/user-attachments/assets/5aaf034e-294b-403c-b3cc-d48ceae423f0)

