import unittest
from cugrad.tensor import Tensor
import math

class TestGradients(unittest.TestCase):
    def test_gradient_addition(self):
        a = Tensor([1])
        a.data = [2.0]
        b = Tensor([1])
        b.data = [3.0]
        c = a + b
        c.backward()
        self.assertEqual(a.grad, [1.0])
        self.assertEqual(b.grad, [1.0])

    def test_gradient_multiplication(self):
        a = Tensor([1])
        a.data = [2.0]
        b = Tensor([1])
        b.data = [3.0]
        c = a * b
        c.backward()
        self.assertEqual(a.grad, [3.0])
        self.assertEqual(b.grad, [2.0])

    def test_gradient_tanh(self):
        a = Tensor([1])
        a.data = [0.0]
        c = a.tanh()
        c.backward()
        # derivative of tanh at 0 is 1.0
        self.assertAlmostEqual(a.grad[0], 1.0 - math.tanh(0.0)**2)

    def test_gradient_mlp(self):
        # Assume MLP now handles arrays properly.
        from cugrad.nn import MLP
        model = MLP(input_size=2, layer_sizes=[2, 1])

        # Create dummy input and target
        inputs = Tensor([2])
        inputs.data = [1.0, 2.0]
        target = Tensor([1])
        target.data = [1.0]

        output = model(inputs)  # output should be shape [1]
        loss = (output - target)*(output - target)
        loss.backward()

        # Check that some parameter grads are non-zero
        for param in model.parameters():
            # just ensure not all zero
            self.assertTrue(any(g != 0.0 for g in param.grad))

if __name__ == "__main__":
    unittest.main()
