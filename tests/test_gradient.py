# tests/test_gradients.py

import unittest
from cugrad.tensor import Tensor
import math

class TestGradients(unittest.TestCase):
    def test_gradient_addition(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a + b
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_gradient_multiplication(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

    def test_gradient_tanh(self):
        a = Tensor(0.0)
        c = a.tanh()
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0 - math.tanh(0.0) ** 2)
        self.assertNotEqual(a.grad, 0.0)

    def test_gradient_mlp(self):
        # Simple test to ensure MLP parameters receive gradients
        from cugrad.nn import MLP
        # Initialize MLP
        model = MLP(input_size=2, layer_sizes=[2, 1])
        # Create dummy input and target
        inputs = [Tensor(1.0), Tensor(2.0)]
        target = Tensor(1.0)
        # Forward pass
        output = model(inputs)[0]
        # Compute loss (MSE)
        loss = (output - target) * (output - target)
        # Backward pass
        loss.backward()
        # Check that gradients are non-zero
        for param in model.parameters():
            print(param.grad)
            self.assertNotEqual(param.grad, 0.0)

if __name__ == "__main__":
    unittest.main()
