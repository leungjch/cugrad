import unittest
import cugrad

class TestValue(unittest.TestCase):
    def test_initialization(self):
        v = cugrad.Tensor()
        self.assertEqual(v.data, 0.0)
        self.assertEqual(v.grad, 0.0)
        self.assertEqual(v._op, ' ')

        v = cugrad.Tensor(5.0)
        self.assertEqual(v.data, 5.0)
        self.assertEqual(v.grad, 0.0)
        self.assertEqual(v._op, ' ')

        v = cugrad.Tensor(3.0, 1.5, '+')
        self.assertEqual(v.data, 3.0)
        self.assertEqual(v.grad, 1.5)
        self.assertEqual(v._op, '+')

    def test_addition(self):
        a = cugrad.Tensor(2.0)
        b = cugrad.Tensor(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '+')

    def test_subtraction(self):
        a = cugrad.Tensor(5.0)
        b = cugrad.Tensor(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '-')

    def test_multiplication(self):
        a = cugrad.Tensor(2.0)
        b = cugrad.Tensor(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '*')

    def test_division(self):
        a = cugrad.Tensor(6.0)
        b = cugrad.Tensor(3.0)
        c = a / b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '/')

    def test_repr(self):
        v = cugrad.Tensor(2.0, 1.0, '+')
        self.assertEqual(repr(v), "<Tensor data=2.000000, grad=1.000000>")

if __name__ == "__main__":
    unittest.main()
