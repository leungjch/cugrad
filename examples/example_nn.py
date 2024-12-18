# example_usage.py

from cugrad.nn import Layer, MLP
from cugrad.tensor import Tensor

model = MLP(2, [16, 16, 1])
print(model)
