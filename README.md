# cugrad

[![Build and Test](https://github.com/leungjch/cugrad/actions/workflows/workflow.yml/badge.svg)](https://github.com/leungjch/cugrad/actions/workflows/workflow.yml)

cugrad is a simple automatic differentiation library written in C++ with Python bindings. It supports CPU and CUDA backends.

Highly inspired and adapted from [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). This is mainly a learning project to understand how automatic differentiation and GPU acceleration works from scratch and not intended for production use.

## Building

To build the cugrad Python bindings, run
```bash.
pip install .
```

